"""Release gate: new eval runs against a committed baseline summary (no LLM calls here).

  # once, from the stored v1 baseline runs -> evals/baselines/v1.json (committed)
  docker compose exec api python -m evals.gate export --runs 5 7 8 9
  # after running the suites for a candidate
  docker compose exec api python -m evals.gate check --runs 10 11 12

FAIL (exit 1): any critical violation; any scenario whose outcome match rate dropped; a
baseline scenario of a suite that ran but has no result (incomplete gate).
WARNING (exit 0): a judge average dropped by more than JUDGE_WARN_DROP; the near-miss rate rose.
Refused (exit 2): the runs are not comparable with the baseline (see evals.compare).
"""

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.config import load_settings
from app.db.session import make_engine
from evals.compare import (
    RULES_CHANGE_PROOF,
    SAME_FIELDS,
    Cell,
    ComparisonError,
    RulesChange,
    ScenarioKey,
    Summary,
    check_comparable,
    parse_rules_change,
    reject_aborted,
    rules_change_used,
    scenario_cells,
    simulated_suites,
    summarize,
)
from evals.report import JUDGE_METRICS, load

BASELINE_DIR = Path(__file__).resolve().parent / "baselines"
# k=1 judge scores are noisy: a WARNING until Day 7 k=3 runs show the real variance.
JUDGE_WARN_DROP = 0.3
RUN_FIELDS = ("id", "suite", "agent_version", "status", "config_hash", "prompt_version",
              "rescored_from_run_id", "simulator_version", *SAME_FIELDS)  # fmt: skip
HASH_NOTE = (
    "Config hashes of these runs predate the Day 4 agent-version restructure (prompts moved "
    "behind agent_versions/*.yaml switches), so a v1 re-run now has a different hash. v1 "
    "behavior is verified identical by tests/test_agent_versions.py::test_v1_prompts_are_unchanged."
)


@dataclass
class GateResult:
    agent_version: str
    baseline_version: str
    run_ids: list[int]
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    summary: tuple[Summary, Summary] | None = None

    @property
    def verdict(self) -> str:
        return "FAIL" if self.failures else "WARNING" if self.warnings else "PASS"


# --- Baseline file ------------------------------------------------------------------------


def _run_meta(run: Any) -> dict[str, Any]:
    return {name: getattr(run, name) for name in RUN_FIELDS} | {"suite": str(run.suite)}


def baseline_doc(runs: list[Any], scores: list[Any], exported_at: datetime) -> dict[str, Any]:
    reject_aborted(runs)
    versions = {r.agent_version for r in runs}
    if len(versions) != 1:
        raise ComparisonError(f"baseline runs mix agent versions: {sorted(versions)}")
    cells, skipped = scenario_cells(runs, scores)
    return {
        "label": "SYNTHETIC",
        "agent_version": versions.pop(),
        "exported_at": exported_at.isoformat(),
        "notes": [HASH_NOTE],
        "runs": [_run_meta(r) for r in sorted(runs, key=lambda r: r.id)],
        "scenarios": [
            {"suite": suite, "scenario_id": sid, **asdict(cell)}
            for (suite, sid), cell in sorted(cells.items())
        ],
        "not_included": skipped,
    }


def read_baseline(doc: dict[str, Any]) -> tuple[list[Any], dict[ScenarioKey, Cell]]:
    runs = [SimpleNamespace(**r) for r in doc["runs"]]
    cells = {}
    for s in doc["scenarios"]:
        fields = {k: v for k, v in s.items() if k not in ("suite", "scenario_id")}
        cells[(s["suite"], s["scenario_id"])] = Cell(**fields | {"run_ids": tuple(s["run_ids"])})
    return runs, cells


# --- Gate ---------------------------------------------------------------------------------


def evaluate(
    doc: dict[str, Any],
    runs: list[Any],
    scores: list[Any],
    simulated: set[str],
    allow_rules_change: RulesChange | None = None,
) -> GateResult:
    base_runs, base_cells = read_baseline(doc)
    check_comparable(base_runs + runs, simulated, allow_rules_change)
    versions = {r.agent_version for r in runs}
    if len(versions) != 1:
        raise ComparisonError(f"candidate runs mix agent versions: {sorted(versions)}")
    cells, skipped = scenario_cells(runs, scores)
    result = GateResult(versions.pop(), doc["agent_version"], sorted(r.id for r in runs))

    if change := rules_change_used(base_runs + runs, allow_rules_change):
        result.notes.append(
            f"rules differ: `{change[0]}` → `{change[1]}`, allowed explicitly with "
            f"`--allow-rules-change` (older versions' decisions proven identical: "
            f"{RULES_CHANGE_PROOF})"
        )
    ran = {str(r.suite) for r in runs}
    for key in sorted(base_cells):
        if key[0] in ran and key not in cells:
            result.failures.append(f"`{key[1]}`: no result (incomplete gate)")
    result.notes += [f"skipped: {s}" for s in skipped]
    result.notes += [f"suite `{s}` not run" for s in sorted({k[0] for k in base_cells} - ran)]

    matched = sorted(base_cells.keys() & cells.keys())
    for key in matched:
        a, b = base_cells[key], cells[key]
        if b.critical:
            result.failures.append(f"`{key[1]}`: {b.critical} critical violation(s)")
        if b.outcome_pass / b.runs < a.outcome_pass / a.runs:
            result.failures.append(
                f"`{key[1]}`: outcome match dropped {a.outcome_pass}/{a.runs} → "
                f"{b.outcome_pass}/{b.runs}"
            )
    base = summarize(base_cells[k] for k in matched)
    new = summarize(cells[k] for k in matched)
    result.summary = (base, new)
    for j in JUDGE_METRICS:
        x, y = base.judge[j], new.judge[j]
        if x is not None and y is not None and x - y > JUDGE_WARN_DROP:
            result.warnings.append(f"{j.removeprefix('judge_')} average {x:.2f} → {y:.2f}")
    if new.near_miss_rate > base.near_miss_rate:
        result.warnings.append(
            f"near-miss rate {base.near_miss_rate:.1%} → {new.near_miss_rate:.1%}"
        )
    return result


def render(r: GateResult) -> str:
    lines = [
        f"# dueflow release gate: **{r.verdict}**",
        "",
        "**Label: SYNTHETIC** — simulated debtors; measures agent behavior, not collection "
        "results. No production outcomes exist.",
        "",
        f"Candidate `{r.agent_version}` (eval runs {', '.join(f'#{i}' for i in r.run_ids)}) "
        f"against the committed `{r.baseline_version}` baseline.",
        "",
    ]
    if r.summary:
        a, b = r.summary
        lines += [
            f"n = {b.scenarios} scenarios · outcome match {a.outcome_pass}/{a.runs} → "
            f"{b.outcome_pass}/{b.runs} · critical {a.critical} → {b.critical} · near-miss "
            f"rate {a.near_miss_rate:.1%} → {b.near_miss_rate:.1%}",
            "",
        ]
    for title, items in (("FAIL", r.failures), ("WARNING", r.warnings), ("Notes", r.notes)):
        lines += [f"## {title}", ""] + ([f"- {i}" for i in items] or ["None."]) + [""]
    return "\n".join(lines)


# --- CLI ----------------------------------------------------------------------------------


async def _load_runs(run_ids: list[int]) -> tuple[list[Any], list[Any]]:
    engine = make_engine(load_settings().app_url)
    try:
        runs, scores = await load(engine, run_ids)
    finally:
        await engine.dispose()
    if missing := sorted(set(run_ids) - {r.id for r in runs}):
        raise ComparisonError(f"unknown eval runs: {missing}")
    return runs, scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="write the baseline summary from stored runs")
    export.add_argument("--runs", type=int, nargs="+", required=True)
    check = sub.add_parser("check", help="gate new runs against the baseline")
    check.add_argument("--runs", type=int, nargs="+", required=True)
    check.add_argument("--report-dir", type=Path, default=Path("reports"))
    check.add_argument("--allow-rules-change", type=parse_rules_change, metavar="OLD:NEW")
    for p in (export, check):
        p.add_argument("--baseline", type=Path, default=BASELINE_DIR / "v1.json")
    args = parser.parse_args()
    try:
        runs, scores = asyncio.run(_load_runs(args.runs))
        if args.command == "export":
            doc = baseline_doc(runs, scores, datetime.now(UTC))
            args.baseline.parent.mkdir(parents=True, exist_ok=True)
            args.baseline.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            print(f"baseline: {args.baseline} ({len(doc['scenarios'])} scenarios)")
            return 0
        doc = json.loads(args.baseline.read_text(encoding="utf-8"))
        result = evaluate(doc, runs, scores, simulated_suites(), args.allow_rules_change)
    except ComparisonError as exc:
        print(f"gate refused: {exc}")
        return 2
    args.report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    path = args.report_dir / f"gate-{result.agent_version}-{stamp}.md"
    path.write_text(render(result), encoding="utf-8")
    print(f"gate: {result.verdict} ({len(result.failures)} failures, "
          f"{len(result.warnings)} warnings) report: {path}")  # fmt: skip
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
