"""Champion vs challenger, built only from stored eval rows.

  docker compose exec api python -m evals.compare --champion 5 7 8 9 --challenger 10 11 12

Each side is one agent version, given as one or more eval runs (a suite may be split over
runs, e.g. a resumed adaptive run). Runs are compared only when everything except the agent
is the same: agent model, judge, rubric, rules, dataset and k, plus the simulator for suites
whose debtor is simulated. Otherwise the comparison is refused, never silently made.
"""

import argparse
import asyncio
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from app.agent.versions import load_version
from app.config import load_settings
from app.db.models import EvalSuite
from app.db.session import make_engine
from evals.report import JUDGE_METRICS, load
from evals.scenario import Kind, load_suite_scenarios

# Must match within a suite across both sides. simulator_version only where it is used.
SAME_FIELDS = (
    "agent_model_id",
    "judge_model_id",
    "rubric_version",
    "rules_version",
    "dataset_hash",
    "k",
)
Verdict = Literal["win", "loss", "tie"]
# An explicit, reported exception to "same rules": (old, new), e.g. ("rules-v6", "rules-v7").
# Only for a rules change proven not to alter the older versions' decisions
# (tests/test_rules_versions.py); the strict same-rules comparison happens at Day 7 (k=3).
RulesChange = tuple[str, str]
RULES_CHANGE_PROOF = "tests/test_rules_versions.py"
ScenarioKey = tuple[str, str]  # (suite, scenario_id)


class ComparisonError(ValueError):
    pass


@dataclass(frozen=True)
class Cell:
    """One scenario on one side, over its k runs."""

    runs: int
    outcome_pass: int
    critical: int
    near_misses: int
    output_checks: int
    judge: dict[str, list[float]]
    agent_tokens: list[float]
    run_ids: tuple[int, ...]


@dataclass(frozen=True)
class Summary:
    scenarios: int
    runs: int
    outcome_pass: int
    critical: int
    near_misses: int
    output_checks: int
    judge: dict[str, float | None]
    agent_tokens_avg: float | None

    @property
    def near_miss_rate(self) -> float:
        return self.near_misses / self.output_checks if self.output_checks else 0.0


@dataclass(frozen=True)
class ScenarioResult:
    suite: str
    scenario_id: str
    champion: Cell
    challenger: Cell
    verdict: Verdict
    regression: bool


@dataclass
class Comparison:
    champion_version: str
    challenger_version: str
    champion_runs: list[int]
    challenger_runs: list[int]
    k: int
    changes: list[str]
    suites: list[str]
    summary: dict[str, tuple[Summary, Summary]]  # suite or "all" -> (champion, challenger)
    scenarios: list[ScenarioResult]
    not_compared: list[str] = field(default_factory=list)
    rules_change: RulesChange | None = None  # set only when the allowed change was used


def simulated_suites() -> set[str]:
    """Suites with a simulated debtor: their simulator version must match too."""
    return {
        suite.value
        for suite in EvalSuite
        if any(s.kind is not Kind.SCRIPTED for s in load_suite_scenarios(suite))
    }


def _version_of(runs: list[Any], side: str) -> str:
    versions = {r.agent_version for r in runs}
    if len(versions) != 1:
        raise ComparisonError(f"{side} runs mix agent versions: {sorted(versions)}")
    return str(versions.pop())


def reject_aborted(runs: list[Any]) -> None:
    """Aborted runs stay on record (e.g. a defect found mid-run) but are never compared."""
    for r in runs:
        if r.status == "aborted":
            reason = getattr(r, "status_reason", None) or "no reason recorded"
            raise ComparisonError(f"run #{r.id} is aborted ({reason}) and is never compared")


def parse_rules_change(value: str) -> RulesChange:
    old, sep, new = value.partition(":")
    if not (sep and old and new and old != new):
        raise ValueError("expected OLD:NEW, e.g. rules-v6:rules-v7")
    return old, new


def rules_change_used(runs: list[Any], allow: RulesChange | None) -> RulesChange | None:
    return allow if allow and len({r.rules_version for r in runs}) > 1 else None


def check_comparable(
    runs: list[Any], simulated: set[str], allow_rules_change: RulesChange | None = None
) -> None:
    reject_aborted(runs)
    by_suite: dict[str, list[Any]] = defaultdict(list)
    for r in runs:
        by_suite[r.suite].append(r)
    for suite, suite_runs in sorted(by_suite.items()):
        fields = SAME_FIELDS + (("simulator_version",) if suite in simulated else ())
        for name in fields:
            values = {getattr(r, name) for r in suite_runs}
            if name == "rules_version" and allow_rules_change and values <= set(allow_rules_change):
                continue
            if len(values) > 1:
                detail = ", ".join(f"#{r.id}={getattr(r, name)}" for r in suite_runs)
                raise ComparisonError(f"suite {suite}: {name} differs ({detail})")


def _changes(champion: str, challenger: str) -> list[str]:
    """Fix switches that differ between the two version files."""
    a, b = load_version(champion).fixes.model_dump(), load_version(challenger).fixes.model_dump()
    return [name for name in b if a.get(name) != b[name]]


def scenario_cells(runs: list[Any], scores: list[Any]) -> tuple[dict[ScenarioKey, Cell], list[str]]:
    """Per scenario, the latest run that holds it; rows with infrastructure errors are skipped."""
    suite_of = {r.id: r.suite for r in runs}
    metrics: dict[tuple[int, str, int], dict[str, Any]] = defaultdict(dict)
    for sc in scores:
        if sc.eval_run_id in suite_of:
            metrics[(sc.eval_run_id, sc.scenario_id, sc.run_index)][sc.metric] = sc
    latest: dict[ScenarioKey, int] = {}
    skipped: list[str] = []
    for (run_id, sid, _), m in sorted(metrics.items()):
        if "infra_error" in m and m["infra_error"].value:
            skipped.append(f"{sid} (run #{run_id}: infrastructure error)")
            continue
        key = (suite_of[run_id], sid)
        latest[key] = max(latest.get(key, run_id), run_id)

    cells: dict[ScenarioKey, Cell] = {}
    for key, run_id in latest.items():
        rows = [
            m
            for (rid, sid, _), m in metrics.items()
            if rid == run_id
            and sid == key[1]
            and not ("infra_error" in m and m["infra_error"].value)
        ]
        cells[key] = Cell(
            runs=len(rows),
            outcome_pass=sum(1 for m in rows if "outcome_match" in m and m["outcome_match"].passed),
            critical=_total(rows, "critical_violations"),
            near_misses=_total(rows, "near_misses"),
            output_checks=_total(rows, "output_checks"),
            judge={j: _values(rows, j) for j in JUDGE_METRICS},
            agent_tokens=_values(rows, "agent_tokens"),
            run_ids=(run_id,),
        )
    return cells, skipped


def _values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    return [float(m[metric].value) for m in rows if metric in m]


def _total(rows: list[dict[str, Any]], metric: str) -> int:
    return int(sum(_values(rows, metric)))


def _verdict(champion: Cell, challenger: Cell) -> tuple[Verdict, bool]:
    """Outcome match rate first, then critical violations. Judge scores never decide (k=1 noise).
    A regression is a loss, or a critical violation the champion did not have."""
    a = champion.outcome_pass / champion.runs
    b = challenger.outcome_pass / challenger.runs
    verdict: Verdict
    if b != a:
        verdict = "win" if b > a else "loss"
    elif challenger.critical != champion.critical:
        verdict = "win" if challenger.critical < champion.critical else "loss"
    else:
        verdict = "tie"
    regression = verdict == "loss" or (challenger.critical > 0 and champion.critical == 0)
    return verdict, regression


def summarize(cells: Iterable[Cell]) -> Summary:
    cells = list(cells)
    tokens = [t for c in cells for t in c.agent_tokens]
    judge: dict[str, float | None] = {}
    for j in JUDGE_METRICS:
        values = [v for c in cells for v in c.judge[j]]
        judge[j] = mean(values) if values else None
    return Summary(
        scenarios=len(cells),
        runs=sum(c.runs for c in cells),
        outcome_pass=sum(c.outcome_pass for c in cells),
        critical=sum(c.critical for c in cells),
        near_misses=sum(c.near_misses for c in cells),
        output_checks=sum(c.output_checks for c in cells),
        judge=judge,
        agent_tokens_avg=mean(tokens) if tokens else None,
    )


def build(
    champion_runs: list[Any],
    challenger_runs: list[Any],
    scores: list[Any],
    simulated: set[str],
    allow_rules_change: RulesChange | None = None,
) -> Comparison:
    if not champion_runs or not challenger_runs:
        raise ComparisonError("each side needs at least one eval run")
    champion = _version_of(champion_runs, "champion")
    challenger = _version_of(challenger_runs, "challenger")
    if champion == challenger:
        raise ComparisonError(f"both sides are agent version {champion}")
    check_comparable(champion_runs + challenger_runs, simulated, allow_rules_change)

    a_cells, a_skipped = scenario_cells(champion_runs, scores)
    b_cells, b_skipped = scenario_cells(challenger_runs, scores)
    not_compared = [f"champion: {s}" for s in a_skipped] + [f"challenger: {s}" for s in b_skipped]
    for key in sorted(a_cells.keys() ^ b_cells.keys()):
        side = "challenger" if key in a_cells else "champion"
        not_compared.append(f"{key[1]} (no result on the {side} side)")

    results = []
    for key in sorted(a_cells.keys() & b_cells.keys()):
        verdict, regression = _verdict(a_cells[key], b_cells[key])
        results.append(
            ScenarioResult(key[0], key[1], a_cells[key], b_cells[key], verdict, regression)
        )
    suites = sorted({r.suite for r in results})
    summary = {
        suite: (
            summarize(r.champion for r in results if r.suite == suite),
            summarize(r.challenger for r in results if r.suite == suite),
        )
        for suite in suites
    }
    summary["all"] = (
        summarize(r.champion for r in results),
        summarize(r.challenger for r in results),
    )
    return Comparison(
        champion_version=champion,
        challenger_version=challenger,
        champion_runs=sorted(r.id for r in champion_runs),
        challenger_runs=sorted(r.id for r in challenger_runs),
        k=int(champion_runs[0].k),
        changes=_changes(champion, challenger),
        suites=suites,
        summary=summary,
        scenarios=results,
        not_compared=not_compared,
        rules_change=rules_change_used(champion_runs + challenger_runs, allow_rules_change),
    )


# --- Markdown -----------------------------------------------------------------------------


def _num(value: float | None, digits: int = 2) -> str:
    return "–" if value is None else f"{value:,.{digits}f}"


def _delta(a: float | None, b: float | None, digits: int = 2) -> str:
    return "–" if a is None or b is None else f"{b - a:+,.{digits}f}"


def _summary_rows(a: Summary, b: Summary) -> list[str]:
    rows = [
        ("Outcome match", f"{a.outcome_pass}/{a.runs}", f"{b.outcome_pass}/{b.runs}",
         f"{b.outcome_pass - a.outcome_pass:+d}"),
        ("Critical violations", str(a.critical), str(b.critical), f"{b.critical - a.critical:+d}"),
        ("Near-miss rate", f"{a.near_miss_rate:.1%} ({a.near_misses}/{a.output_checks})",
         f"{b.near_miss_rate:.1%} ({b.near_misses}/{b.output_checks})",
         f"{(b.near_miss_rate - a.near_miss_rate) * 100:+.1f} pts"),
    ]  # fmt: skip
    for j in JUDGE_METRICS:
        label = "Judge " + j.removeprefix("judge_").replace("_", "-")
        rows.append((label, _num(a.judge[j]), _num(b.judge[j]), _delta(a.judge[j], b.judge[j])))
    rows.append(
        (
            "Agent tokens / conversation",
            _num(a.agent_tokens_avg, 0),
            _num(b.agent_tokens_avg, 0),
            _delta(a.agent_tokens_avg, b.agent_tokens_avg, 0),
        )
    )
    return [f"| {m} | {x} | {y} | {d} |" for m, x, y, d in rows]


def _judge_mean(cell: Cell) -> str:
    values = [v for vs in cell.judge.values() for v in vs]
    return _num(mean(values)) if values else "–"


def render(c: Comparison) -> str:
    a, b = c.champion_version, c.challenger_version
    wins = sum(1 for r in c.scenarios if r.verdict == "win")
    losses = sum(1 for r in c.scenarios if r.verdict == "loss")
    ties = len(c.scenarios) - wins - losses
    lines = [
        f"# dueflow champion vs challenger: `{a}` vs `{b}`",
        "",
        "**Label: SYNTHETIC** — simulated debtors; measures agent behavior, not collection "
        "results. No production outcomes exist.",
        "",
        f"**`{b}` is measured as one combined change** of {len(c.changes)} fixes: "
        + ", ".join(f"`{x}`" for x in c.changes)
        + ". These numbers do not say which fix caused which difference.",
        "",
        "| | |",
        "|---|---|",
        f"| Champion | `{a}`, eval runs {', '.join(f'#{i}' for i in c.champion_runs)} |",
        f"| Challenger | `{b}`, eval runs {', '.join(f'#{i}' for i in c.challenger_runs)} |",
        f"| n | {len(c.scenarios)} scenarios × k={c.k} per side |",
        f"| Scenario verdicts | {wins} wins · {losses} losses · {ties} ties |",
        "",
    ]
    if c.rules_change:
        old, new = c.rules_change
        lines += [
            f"**Rules differ: `{old}` → `{new}`**, allowed explicitly with "
            f"`--allow-rules-change {old}:{new}`. The older versions' policy decisions are proven "
            f"identical under both ({RULES_CHANGE_PROOF}). The strict same-rules comparison is "
            "Day 7 (k=3).",
            "",
        ]
    if c.k == 1:
        lines += [
            "k=1: one conversation per scenario. Judge averages move by chance at this size; "
            "read them as direction, not proof.",
            "",
        ]
    for suite in [*c.suites, "all"]:
        sa, sb = c.summary[suite]
        title = "All suites" if suite == "all" else f"Suite `{suite}`"
        lines += [
            f"## {title} (n={sa.scenarios})",
            "",
            f"| Metric | `{a}` | `{b}` | Δ |",
            "|---|---|---|---|",
            *_summary_rows(sa, sb),
            "",
        ]
    regressions = [r for r in c.scenarios if r.regression]
    lines += ["## Regressions", ""]
    lines += [
        f"- `{r.scenario_id}`: outcome {r.champion.outcome_pass}/{r.champion.runs} → "
        f"{r.challenger.outcome_pass}/{r.challenger.runs}, critical {r.champion.critical} → "
        f"{r.challenger.critical}"
        for r in regressions
    ] or ["None."]
    lines += [
        "",
        "## Per scenario",
        "",
        f"| Scenario | Verdict | Outcome `{a}` → `{b}` | Critical | Near misses "
        "| Judge mean | Runs |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in c.scenarios:
        x, y = r.champion, r.challenger
        lines.append(
            f"| `{r.scenario_id}` | {r.verdict}{' ⚠️' if r.regression else ''} "
            f"| {x.outcome_pass}/{x.runs} → {y.outcome_pass}/{y.runs} "
            f"| {x.critical} → {y.critical} | {x.near_misses} → {y.near_misses} "
            f"| {_judge_mean(x)} → {_judge_mean(y)} "
            f"| #{x.run_ids[0]} → #{y.run_ids[0]} |"
        )
    if c.not_compared:
        lines += ["", "## Not compared", ""] + [f"- {s}" for s in c.not_compared]
    lines.append("")
    return "\n".join(lines)


# --- CLI ----------------------------------------------------------------------------------


async def compare(
    champion: list[int], challenger: list[int], allow_rules_change: RulesChange | None = None
) -> Comparison:
    engine = make_engine(load_settings().app_url)
    try:
        runs, scores = await load(engine, champion + challenger)
    finally:
        await engine.dispose()
    found = {r.id for r in runs}
    if missing := sorted(set(champion + challenger) - found):
        raise ComparisonError(f"unknown eval runs: {missing}")
    return build(
        [r for r in runs if r.id in champion],
        [r for r in runs if r.id in challenger],
        scores,
        simulated_suites(),
        allow_rules_change,
    )


def write(c: Comparison, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    path = out_dir / f"compare-{c.champion_version}-{c.challenger_version}-{stamp}.md"
    path.write_text(render(c), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", type=int, nargs="+", required=True)
    parser.add_argument("--challenger", type=int, nargs="+", required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--allow-rules-change", type=parse_rules_change, metavar="OLD:NEW",
        help="compare across this one rules change (printed in the report)",
    )  # fmt: skip
    args = parser.parse_args()
    if overlap := set(args.champion) & set(args.challenger):
        parser.error(f"runs on both sides: {sorted(overlap)}")
    try:
        result = asyncio.run(compare(args.champion, args.challenger, args.allow_rules_change))
    except ComparisonError as exc:
        raise SystemExit(f"comparison refused: {exc}") from exc
    print(f"report: {write(result, args.report_dir)}")
