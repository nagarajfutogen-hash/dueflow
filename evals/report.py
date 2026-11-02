"""Markdown report built only from stored eval rows (no number is computed elsewhere)."""

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.models import EvalRun, EvalScore

JUDGE_METRICS = ("judge_tone", "judge_empathy", "judge_clarity", "judge_de_escalation")
TABLE_HEADER = (
    "| Scenario | Expected | Actual | Outcome | Critical | Near misses "
    "| Tone | Empathy | Clarity | De-esc. | Agent tokens |"
)
Rows = dict[tuple[int, str, int], dict[str, Any]]


async def load(engine: AsyncEngine, run_ids: list[int]) -> tuple[list[Any], list[Any]]:
    async with engine.connect() as conn:
        runs = list(
            await conn.execute(
                select(EvalRun.__table__).where(EvalRun.id.in_(run_ids)).order_by(EvalRun.id)
            )
        )
        scores = list(
            await conn.execute(
                select(EvalScore.__table__)
                .where(EvalScore.eval_run_id.in_(run_ids))
                .order_by(EvalScore.id)
            )
        )
    return runs, scores


def _sum(rows: Rows, metric: str) -> int:
    return sum(int(m[metric].value) for m in rows.values() if metric in m)


def _passed(rows: Rows, metric: str) -> int:
    return sum(1 for m in rows.values() if metric in m and m[metric].passed)


def _fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:,.0f}"


def _header(run: Any, runs: list[Any]) -> list[str]:
    run_list = ", ".join(
        f"#{r.id} {r.suite} ({r.status}, k={r.k}, {r.simulator_version}, {r.rubric_version}"
        + (f", rescore of #{r.rescored_from_run_id}" if r.rescored_from_run_id else "")
        + ")"
        for r in runs
    )
    return [
        "# dueflow eval report",
        "",
        "**Label: SYNTHETIC** — simulated debtors; measures agent behavior, not collection "
        "results. No production outcomes exist.",
        "",
        "| | |",
        "|---|---|",
        f"| Agent version | `{run.agent_version}` |",
        f"| Agent model | `{run.agent_model_id}` |",
        f"| Simulator | `{run.simulator_model_id}` (`{run.simulator_version}`) |",
        f"| Judge model | `{run.judge_model_id}` (rubric `{run.rubric_version}`) |",
        f"| Rules / prompts / scripts | `{run.rules_version}` / `{run.prompt_version}` / "
        f"`{run.scripts_version}` |",
        f"| Config hash | `{run.config_hash[:12]}` |",
        f"| Eval runs | {run_list} |",
        "",
    ]


def _scenario_row(scenario_id: str, idx: int, m: dict[str, Any]) -> str:
    om = m.get("outcome_match")
    expected = ", ".join(om.evidence.get("expected", [])) if om else "–"
    actual = om.evidence.get("actual") if om else "–"
    crit = m.get("critical_violations")
    crit_cell = ("❌ " if crit and crit.value else "") + _fmt(crit.value if crit else None)
    judge = " | ".join(_fmt(m[j].value if j in m else None) for j in JUDGE_METRICS)
    near = _fmt(m["near_misses"].value) if "near_misses" in m else "–"
    tokens = _fmt(m["agent_tokens"].value) if "agent_tokens" in m else "–"
    name = f"`{scenario_id}`" + ("" if idx == 0 else f" #{idx}")
    ok = "✅" if om and om.passed else "❌"
    return f"| {name} | {expected} | {actual} | {ok} | {crit_cell} | {near} | {judge} | {tokens} |"


def render(runs: list[Any], scores: list[Any]) -> str:
    by_scenario: Rows = defaultdict(dict)
    for sc in scores:
        by_scenario[(sc.eval_run_id, sc.scenario_id, sc.run_index)][sc.metric] = sc

    lines = _header(runs[0], runs)
    total_critical = near_total = checks_total = 0
    for run in runs:
        rows = {key: m for key, m in by_scenario.items() if key[0] == run.id}
        n = len(rows)
        critical = _sum(rows, "critical_violations")
        near = _sum(rows, "near_misses")
        checks = _sum(rows, "output_checks")
        total_critical += critical
        near_total += near
        checks_total += checks
        lines += [
            f"## Suite `{run.suite}` — run #{run.id}",
            "",
            f"- Scenarios: **{n}** · outcome match **{_passed(rows, 'outcome_match')}/{n}** · "
            f"critical violations **{critical}** · near misses **{near}** of {checks} "
            "output checks",
            "",
            TABLE_HEADER,
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        lines += [_scenario_row(sid, idx, m) for (_, sid, idx), m in sorted(rows.items())]
        lines.append("")

    lines += ["## Critical violations", ""]
    found = False
    for (run_id, scenario_id, _), m in sorted(by_scenario.items()):
        crit = m.get("critical_violations")
        for item in crit.evidence.get("items", []) if crit else []:
            found = True
            lines.append(
                f"- `{scenario_id}` (run #{run_id}): **{item['kind']}** — {item['detail']} "
                f"`{item.get('evidence', {})}`"
            )
    if not found:
        lines.append("None. ✅")

    lines += ["", "## Near misses (interceptor blocks: the guard worked)", ""]
    found = False
    for (_, scenario_id, _), m in sorted(by_scenario.items()):
        nm = m.get("near_misses")
        for b in nm.evidence.get("blocked", []) if nm else []:
            found = True
            lines.append(f"- `{scenario_id}`: {b.get('code')} ({b.get('detail')})")
    if not found:
        lines.append("None.")
    rate = near_total / checks_total if checks_total else 0.0
    lines += [
        "",
        f"Near-miss rate: **{near_total}/{checks_total} = {rate:.1%}** of agent replies.",
        "",
    ]

    averages = []
    for j in JUDGE_METRICS:
        values = [m[j].value for m in by_scenario.values() if j in m]
        label = j.removeprefix("judge_")
        averages.append(f"{label} {mean(values):.2f}" if values else f"{label} –")
    lines += [
        "## Judge (language quality, 1–5)",
        "",
        "Averages: " + " · ".join(averages),
        f"Judge errors: {_sum(by_scenario, 'judge_error')}",
        "",
        "## Tokens (Groq)",
        "",
        "| Run | Agent | Simulator | Judge |",
        "|---|---|---|---|",
    ]
    grand = 0
    for run in runs:
        cells = []
        for part in ("agent", "simulator", "judge"):
            p = (run.usage or {}).get(part, {})
            tokens = int(p.get("prompt_tokens", 0)) + int(p.get("completion_tokens", 0))
            grand += tokens
            cells.append(f"{tokens:,} ({p.get('calls', 0)} calls)")
        lines.append(f"| #{run.id} {run.suite} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"**Total: {grand:,} tokens.** Critical violations across all suites: "
        f"**{total_critical}**.",
        "",
    ]
    return "\n".join(lines)


def _write(out_dir: Path, name: str, content: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(content, encoding="utf-8")
    return path


async def write_report(engine: AsyncEngine, run_ids: list[int], out_dir: Path) -> Path:
    runs, scores = await load(engine, run_ids)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    name = f"eval-{runs[0].agent_version}-{stamp}-runs-{'-'.join(map(str, run_ids))}.md"
    return await asyncio.to_thread(_write, out_dir, name, render(runs, scores))
