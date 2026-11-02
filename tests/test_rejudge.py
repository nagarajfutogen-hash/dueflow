"""evals.run --rejudge-errors: only judge_error conversations are judged again. No LLM call."""

from typing import Any

import httpx
import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.models import EvalRun, EvalScore, RunStatus
from app.llm.groq_chat import InvalidJSONError
from evals.judge import load_rubric
from evals.run import Context, rejudge_run
from evals.store import record_score
from tests.conftest import MakeDebtor
from tests.test_chat_offline import _session
from tests.test_evals_offline import StubChat, _judge_json
from tests.test_lab import MODELS, RULES, _run

RUBRIC = load_rubric().version  # rejudge refuses a run judged under another rubric


async def _source(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, engine: AsyncEngine
) -> tuple[int, int]:
    session, _, _ = await _session(api_client, make_debtor, engine)
    await session.say_scripted("Hello, this is an AI assistant.")
    await session.record_debtor("Who is this?")
    cid = session.conversation_id
    run_id = await _run(engine, scores=[], conversation_id=cid, rubric_version=RUBRIC)
    rows: list[tuple[str, str, float, bool | None, dict[str, Any]]] = [
        ("base/failed", "outcome_match", 1.0, True, {"actual": "promise"}),
        ("base/failed", "critical_violations", 0.0, True, {"items": []}),
        ("base/failed", "judge_error", 1.0, False, {"error": "unavailable: HTTP 400"}),
        ("base/ok", "outcome_match", 0.0, False, {"actual": "disconnected"}),
        ("base/ok", "critical_violations", 0.0, True, {"items": []}),
        ("base/ok", "judge_tone", 4.0, None, {"reason": "kept"}),
        ("base/ok", "judge_error", 0.0, True, {"error": None}),
    ]
    async with engine.begin() as conn:
        for sid, metric, value, passed, evidence in rows:
            await record_score(
                conn, run_id=run_id, scenario_id=sid, run_index=0, metric=metric, value=value,
                passed=passed, conversation_id=cid, evidence=evidence,
            )  # fmt: skip
    return run_id, cid


def _ctx(engine: AsyncEngine, replies: list[str | Exception]) -> tuple[Context, StubChat]:
    chat = StubChat(replies)
    return Context(engine, RULES, MODELS, load_rubric(), chat, "offline", "v1"), chat  # type: ignore[arg-type]


async def _scores(engine: AsyncEngine, run_id: int) -> dict[tuple[str, str], Any]:
    async with engine.connect() as conn:
        rows = await conn.execute(select(EvalScore).where(EvalScore.eval_run_id == run_id))
        return {(r.scenario_id, r.metric): r for r in rows}


async def test_only_failed_judges_are_rerun_and_the_rest_is_copied(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    source, cid = await _source(api_client, make_debtor, app_engine)
    before = await _scores(app_engine, source)
    ctx, chat = _ctx(app_engine, [_judge_json()])  # exactly one judge call allowed
    new = await rejudge_run(ctx, source)
    assert new is not None and chat.replies == []
    got = await _scores(app_engine, new)
    assert got[("base/failed", "judge_error")].passed is True
    assert ("base/failed", "judge_tone") in got  # the new judge scores
    assert got[("base/failed", "outcome_match")].evidence == {"actual": "promise"}  # copied
    for key in [("base/ok", m) for m in ("outcome_match", "judge_tone", "judge_error")]:
        assert (got[key].value, got[key].passed, got[key].evidence) == (
            before[key].value, before[key].passed, before[key].evidence,
        )  # fmt: skip
    async with app_engine.connect() as conn:
        run = (await conn.execute(select(EvalRun).where(EvalRun.id == new))).one()
        source_rows = await conn.scalar(select(func.count()).where(EvalScore.eval_run_id == source))
    assert run.rescored_from_run_id == source and run.status is RunStatus.COMPLETED
    assert run.usage["agent"]["prompt_tokens"] == 0 and run.usage["judge"]["prompt_tokens"] > 0
    assert source_rows == len(before)  # the source run is never changed


async def test_a_judge_that_fails_again_stays_a_judge_error(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    source, _ = await _source(api_client, make_debtor, app_engine)
    bad = InvalidJSONError("HTTP 400: json_validate_failed")
    ctx, chat = _ctx(app_engine, [bad, bad, bad])
    new = await rejudge_run(ctx, source)
    assert new is not None and chat.replies == []  # 3 attempts, then it gives up
    got = await _scores(app_engine, new)
    assert got[("base/failed", "judge_error")].passed is False


async def test_runs_without_judge_errors_are_skipped(app_engine: AsyncEngine) -> None:
    run_id = await _run(
        app_engine, scores=[("base/x", 0, "judge_error", 0.0, True)], rubric_version=RUBRIC
    )
    ctx, chat = _ctx(app_engine, [])
    assert await rejudge_run(ctx, run_id) is None


async def test_only_original_completed_runs_are_rejudged(app_engine: AsyncEngine) -> None:
    run_id = await _run(
        app_engine, scores=[("base/x", 0, "judge_error", 1.0, False)], rubric_version=RUBRIC
    )
    async with app_engine.begin() as conn:
        await conn.execute(
            update(EvalRun)
            .where(EvalRun.id == run_id)
            .values(status=RunStatus.ABORTED, status_reason="test")
        )
    ctx, _ = _ctx(app_engine, [])
    with pytest.raises(SystemExit, match="not an original completed run"):
        await rejudge_run(ctx, run_id)
