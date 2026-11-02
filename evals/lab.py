"""Failure Lab: failures from stored eval scores, one conversation's full record, promotion of
a failure to the regressions suite, and the token estimate shown before a replay.

Every number comes from stored rows (eval_runs, eval_scores, turns, events). Nothing here
calls an LLM.
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yaml
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncConnection

from app.db.models import (
    Conversation,
    EvalRun,
    EvalScore,
    EvalSuite,
    Event,
    RunStatus,
    Turn,
)
from evals.scenario import REGRESSIONS_DIR, Scenario, find_scenario

# A failed check (passed = false) -> its failure category.
CATEGORIES: dict[str, str] = {
    "critical_violations": "critical_violation",
    "outcome_match": "wrong_outcome",
    "handoff_accuracy": "wrong_handoff",
    "expected_events": "missing_event",
    "promise_completeness": "incomplete_promise",
    "infra_error": "infra_error",
    "judge_error": "judge_error",
}
MAX_FAILURES = 500
TOKEN_METRICS = ("agent_tokens", "debtor_tokens")


def categories(failed_metrics: list[str]) -> list[str]:
    """Failure categories in CATEGORIES order (most severe first); unknown metrics skipped."""
    return [cat for metric, cat in CATEGORIES.items() if metric in failed_metrics]


def regression_id(scenario_id: str) -> str:
    return scenario_id if scenario_id.startswith("regressions/") else f"regressions/{scenario_id}"


def regression_path(scenario_id: str, root: Path = REGRESSIONS_DIR) -> Path:
    slug = re.sub(r"[^a-z0-9_-]+", "__", scenario_id.lower()).strip("_")
    return root / f"{slug}.yaml"


async def hidden_runs(conn: AsyncConnection) -> set[int]:
    """Runs left out of the default list: not completed (aborted / failed / running), or
    superseded by a re-score (the source run, and any older re-score of the same source).
    Each re-score copies its source's failures, so listing every run repeats them."""
    runs = list(
        await conn.execute(select(EvalRun.id, EvalRun.status, EvalRun.rescored_from_run_id))
    )
    hidden = {r.id for r in runs if r.status != RunStatus.COMPLETED}
    latest: dict[int, int] = {}  # source run -> its newest completed re-score
    for r in runs:
        if r.status == RunStatus.COMPLETED and r.rescored_from_run_id is not None:
            src = r.rescored_from_run_id
            latest[src] = max(latest.get(src, r.id), r.id)
    for r in runs:
        src = r.rescored_from_run_id
        if src is not None and r.id != latest.get(src):
            hidden.add(r.id)
    return hidden | set(latest)


async def failures(
    conn: AsyncConnection,
    *,
    category: str | None = None,
    version: str | None = None,
    label: str | None = None,
    history: bool = False,
) -> list[dict[str, Any]]:
    """One row per failed conversation (run, scenario, run_index), newest run first. Without
    `history`, only current runs (see hidden_runs)."""
    query = (
        select(
            EvalScore.eval_run_id,
            EvalScore.scenario_id,
            EvalScore.run_index,
            EvalScore.conversation_id,
            EvalScore.metric,
            EvalRun.suite,
            EvalRun.agent_version,
            EvalRun.label,
            EvalRun.status,
            EvalRun.simulator_version,
            EvalRun.started_at,
        )
        .join(EvalRun, EvalRun.id == EvalScore.eval_run_id)
        .where(EvalScore.passed.is_(False), EvalScore.metric.in_(list(CATEGORIES)))
        .order_by(EvalScore.eval_run_id.desc(), EvalScore.scenario_id, EvalScore.run_index)
    )
    if version:
        query = query.where(EvalRun.agent_version == version)
    if label:
        query = query.where(EvalRun.label == label)
    if not history:
        hidden = await hidden_runs(conn)
        if hidden:
            query = query.where(EvalScore.eval_run_id.not_in(hidden))
    rows: dict[tuple[int, str, int], dict[str, Any]] = {}
    for r in await conn.execute(query):
        key = (r.eval_run_id, r.scenario_id, r.run_index)
        row = rows.setdefault(
            key,
            {
                "eval_run_id": r.eval_run_id,
                "scenario_id": r.scenario_id,
                "run_index": r.run_index,
                "conversation_id": r.conversation_id,
                "suite": r.suite.value,
                "agent_version": r.agent_version,
                "label": r.label.value,
                "run_status": r.status.value,
                "simulator_version": r.simulator_version,
                "started_at": r.started_at.isoformat(),
                "failed_metrics": [],
            },
        )
        row["failed_metrics"].append(r.metric)
    out = []
    for row in rows.values():
        row["categories"] = categories(row.pop("failed_metrics"))
        row["promoted"] = regression_path(regression_id(row["scenario_id"])).exists()
        if category is None or category in row["categories"]:
            out.append(row)
    return out[:MAX_FAILURES]


async def conversation_detail(conn: AsyncConnection, conversation_id: int) -> dict[str, Any] | None:
    c = (
        await conn.execute(select(Conversation).where(Conversation.id == conversation_id))
    ).one_or_none()
    if c is None:
        return None
    turns = await conn.execute(
        select(Turn).where(Turn.conversation_id == conversation_id).order_by(Turn.seq)
    )
    events = await conn.execute(
        select(Event).where(Event.conversation_id == conversation_id).order_by(Event.id)
    )
    scores = await conn.execute(
        select(EvalScore).where(EvalScore.conversation_id == conversation_id).order_by(EvalScore.id)
    )
    score_rows = list(scores)
    failed = [s.metric for s in score_rows if s.passed is False]
    return {
        "conversation": {
            "id": c.id,
            "channel": c.channel.value,
            "label": c.label.value,
            "agent_version": c.agent_version,
            "scenario_id": c.scenario_id,
            "outcome": c.outcome.value if c.outcome else None,
            "sim_started_at": c.sim_started_at.isoformat(),
            "llm_calls": c.llm_calls,
            "prompt_tokens": c.prompt_tokens,
            "completion_tokens": c.completion_tokens,
        },
        "turns": [
            {
                "seq": t.seq,
                "speaker": t.speaker.value,
                "text": t.text,
                "source": "script" if t.llm_model == "script" else ("llm" if t.llm_model else None),
                "prompt_version": t.prompt_version,
                "latency_ms": t.latency_ms,
            }
            for t in turns
        ],
        "events": [
            {
                "id": e.id,
                "type": e.type.value,
                "rule": e.rule,
                "decision": e.decision.value if e.decision else None,
                "reason": e.reason,
                "input": e.input,
                "clock_time": e.clock_time.isoformat() if e.clock_time else None,
            }
            for e in events
        ],
        "scores": [
            {
                "eval_run_id": s.eval_run_id,
                "run_index": s.run_index,
                "metric": s.metric,
                "value": s.value,
                "passed": s.passed,
                "evidence": s.evidence,
            }
            for s in score_rows
        ],
        "categories": categories(failed),
    }


class PromoteError(ValueError):
    pass


async def promote(
    conn: AsyncConnection,
    *,
    eval_run_id: int,
    scenario_id: str,
    run_index: int,
    root: Path = REGRESSIONS_DIR,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Copy the failed scenario into the regressions suite (one YAML file per case). Promoting
    the same scenario again changes nothing: the first failure that promoted it is kept."""
    failed = list(
        await conn.execute(
            select(EvalScore.metric, EvalScore.conversation_id).where(
                EvalScore.eval_run_id == eval_run_id,
                EvalScore.scenario_id == scenario_id,
                EvalScore.run_index == run_index,
                EvalScore.passed.is_(False),
                EvalScore.metric.in_(list(CATEGORIES)),
            )
        )
    )
    if not failed:
        raise PromoteError("no failed check for this run, scenario and run index")
    source = find_scenario(scenario_id)
    if source is None:
        raise PromoteError(f"scenario {scenario_id} is not in any suite file")
    rid = regression_id(scenario_id)  # a failed regression maps onto its own file
    case = source.model_dump(mode="json", exclude_defaults=True)
    case |= {"id": rid, "suite": EvalSuite.REGRESSIONS.value}
    Scenario.model_validate(case)  # must load as a valid regressions scenario
    provenance = {
        "eval_run_id": eval_run_id,
        "scenario_id": scenario_id,
        "run_index": run_index,
        "conversation_id": failed[0].conversation_id,
        "categories": categories([f.metric for f in failed]),
        "promoted_at": (now or datetime.now(UTC)).isoformat(),
    }
    path = regression_path(rid, root)
    created = _write_case(path, {"promoted_from": provenance, "scenarios": [case]})
    return {"regression_id": rid, "file": path.name, "created": created}


def _write_case(path: Path, doc: dict[str, Any]) -> bool:
    """Write the case unless the file exists (sync: one small local file)."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# Promoted from the Failure Lab. SYNTHETIC data. Committed by the owner.\n"
    path.write_text(header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), "utf-8")
    return True


async def token_estimate(conn: AsyncConnection, scenario_id: str) -> dict[str, Any]:
    """Expected tokens for one replay side, from this scenario's stored completed runs:
    agent + simulator tokens per conversation, plus the judge's per-conversation share."""
    rows = list(
        await conn.execute(
            select(EvalScore.eval_run_id, EvalScore.run_index, EvalScore.metric, EvalScore.value)
            .join(EvalRun, EvalRun.id == EvalScore.eval_run_id)
            .where(
                EvalScore.scenario_id == scenario_id,
                EvalScore.metric.in_(TOKEN_METRICS),
                EvalRun.status == RunStatus.COMPLETED,
                EvalRun.rescored_from_run_id.is_(None),
            )
        )
    )
    per_convo: dict[tuple[int, int], float] = {}
    for r in rows:
        per_convo[(r.eval_run_id, r.run_index)] = (
            per_convo.get((r.eval_run_id, r.run_index), 0) + r.value
        )
    run_ids = {run_id for run_id, _ in per_convo}
    judge_share = await _judge_per_conversation(conn, run_ids)
    if not per_convo:
        return {"scenario_id": scenario_id, "n": 0, "per_side": None, "total": None}
    per_side = round(mean(per_convo.values()) + judge_share)
    return {
        "scenario_id": scenario_id,
        "n": len(per_convo),
        "per_side": per_side,
        "total": 2 * per_side,
    }


async def _judge_per_conversation(conn: AsyncConnection, run_ids: set[int]) -> float:
    if not run_ids:
        return 0.0
    convos = dict(
        list(
            await conn.execute(
                select(EvalScore.eval_run_id, func.count())
                .where(EvalScore.eval_run_id.in_(run_ids), EvalScore.metric == "outcome_match")
                .group_by(EvalScore.eval_run_id)
            )
        )
    )
    shares = []
    for run in await conn.execute(select(EvalRun.id, EvalRun.usage).where(EvalRun.id.in_(run_ids))):
        judge = (run.usage or {}).get("judge", {})
        tokens = int(judge.get("prompt_tokens", 0)) + int(judge.get("completion_tokens", 0))
        if convos.get(run.id):
            shares.append(tokens / convos[run.id])
    return mean(shares) if shares else 0.0
