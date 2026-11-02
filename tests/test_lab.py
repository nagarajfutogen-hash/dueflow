"""Failure Lab backend: failures, detail, promote, token estimate, replay. No LLM calls."""

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import insert, update
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.models import EvalRun, EvalSuite, Replay, ReplayStatus, RunStatus
from app.models_config import load_models
from app.policy.config import load_rules
from evals import lab, replay
from evals import scenario as scenario_mod
from evals.scenario import load_suite
from evals.store import finish_run, record_score, start_run
from tests.conftest import MakeDebtor
from tests.test_chat_offline import _session

MODELS = load_models()
RULES = load_rules()
NOW = datetime(2026, 10, 27, 3, tzinfo=UTC)


async def _run(
    engine: AsyncEngine,
    version: str = "v1",
    *,
    suite: EvalSuite = EvalSuite.BASE,
    usage: dict[str, Any] | None = None,
    scores: list[tuple[str, int, str, float, bool | None]] = (),  # type: ignore[assignment]
    conversation_id: int | None = None,
    rubric_version: str = "rubric-v1",
) -> int:
    async with engine.begin() as conn:
        run_id = await start_run(
            conn,
            suite=suite,
            agent_version=version,
            k=1,
            rubric_version=rubric_version,
            dataset_files=[scenario_mod.SUITE_FILES[EvalSuite.BASE]],
            models=MODELS,
            rules=RULES,
            now=NOW,
            seed=42,
        )
        for sid, idx, metric, value, passed in scores:
            await record_score(
                conn,
                run_id=run_id,
                scenario_id=sid,
                run_index=idx,
                metric=metric,
                value=value,
                passed=passed,
                conversation_id=conversation_id,
            )
        await finish_run(conn, run_id, status=RunStatus.COMPLETED, usage=usage or {}, now=NOW)
    return run_id


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _listing(path: Path) -> list[Path]:
    return list(path.iterdir())


def _uid(prefix: str = "base") -> str:
    return f"{prefix}/lab-{uuid.uuid4().hex[:8]}"


# --- Categories and files -------------------------------------------------------------------


def test_categories_follow_the_failed_checks_most_severe_first() -> None:
    assert lab.categories(["outcome_match", "critical_violations", "near_misses"]) == [
        "critical_violation",
        "wrong_outcome",
    ]
    assert lab.categories(["agent_tokens"]) == []


def test_regression_ids_and_files() -> None:
    assert lab.regression_id("base/angry") == "regressions/base/angry"
    assert lab.regression_id("regressions/base/angry") == "regressions/base/angry"
    assert lab.regression_path("regressions/base/angry", Path("/r")).name == (
        "regressions__base__angry.yaml"
    )


# --- Failures list --------------------------------------------------------------------------


async def test_failures_lists_failed_conversations_with_filters(app_engine: AsyncEngine) -> None:
    bad, good = _uid(), _uid()
    run_id = await _run(
        app_engine,
        "v2",
        scores=[
            (bad, 0, "outcome_match", 0.0, False),
            (bad, 0, "critical_violations", 0.0, True),
            (bad, 0, "near_misses", 3.0, None),
            (good, 0, "outcome_match", 1.0, True),
        ],
    )
    async with app_engine.connect() as conn:
        rows = [r for r in await lab.failures(conn) if r["eval_run_id"] == run_id]
        assert [(r["scenario_id"], r["categories"]) for r in rows] == [(bad, ["wrong_outcome"])]
        assert rows[0]["label"] == "SYNTHETIC" and rows[0]["agent_version"] == "v2"
        assert not [r for r in await lab.failures(conn, version="v1") if r["eval_run_id"] == run_id]
        assert not [
            r
            for r in await lab.failures(conn, category="critical_violation")
            if r["eval_run_id"] == run_id
        ]
        assert not [
            r for r in await lab.failures(conn, label="HUMAN_TEST") if r["eval_run_id"] == run_id
        ]


async def test_default_list_hides_superseded_and_unfinished_runs(app_engine: AsyncEngine) -> None:
    sid = _uid()
    failed: list[tuple[str, int, str, float, bool | None]] = [(sid, 0, "judge_error", 1.0, False)]
    source, older, newer, aborted = [await _run(app_engine, scores=failed) for _ in range(4)]
    async with app_engine.begin() as conn:
        for rescore in (older, newer):
            await conn.execute(
                update(EvalRun).where(EvalRun.id == rescore).values(rescored_from_run_id=source)
            )
        await conn.execute(
            update(EvalRun)
            .where(EvalRun.id == aborted)
            .values(status=RunStatus.ABORTED, status_reason="test")
        )
    async with app_engine.connect() as conn:
        current = {r["eval_run_id"] for r in await lab.failures(conn) if r["scenario_id"] == sid}
        everything = {
            r["eval_run_id"]
            for r in await lab.failures(conn, history=True)
            if r["scenario_id"] == sid
        }
    assert current == {newer}  # source and older re-score superseded, aborted run hidden
    assert everything == {source, older, newer, aborted}


async def test_failures_api_passes_history(api_client: httpx.AsyncClient) -> None:
    for query in ("", "?history=true"):
        r = await api_client.get(f"/lab/failures{query}")
        assert r.status_code == 200


# --- Conversation detail --------------------------------------------------------------------


async def test_detail_has_transcript_decisions_and_judge_reasons(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, _ = await _session(api_client, make_debtor, app_engine)
    await session.say_scripted("Hello.")
    await session.note_voice({"ttfb_ms": 800})
    cid = session.conversation_id
    run_id = await _run(app_engine, scores=[], conversation_id=cid)
    async with app_engine.begin() as conn:
        await record_score(
            conn, run_id=run_id, scenario_id="base/x", run_index=0, metric="outcome_match",
            value=0.0, passed=False, conversation_id=cid,
        )  # fmt: skip
        await record_score(
            conn, run_id=run_id, scenario_id="base/x", run_index=0, metric="judge_empathy",
            value=2.0, conversation_id=cid, evidence={"reason": "curt"},
        )  # fmt: skip
    async with app_engine.connect() as conn:
        d = await lab.conversation_detail(conn, cid)
        assert await lab.conversation_detail(conn, 10**9) is None
    assert d is not None
    assert d["conversation"]["label"] in ("SYNTHETIC", "HUMAN_TEST")
    assert d["turns"][-1] == {
        "seq": d["turns"][-1]["seq"], "speaker": "agent", "text": "Hello.", "source": "script",
        "prompt_version": d["turns"][-1]["prompt_version"], "latency_ms": {"ttfb_ms": 800},
    }  # fmt: skip
    rules = {e["rule"] for e in d["events"] if e["type"] == "policy_decision"}
    assert "contact_window" in rules  # the pre-call check, with its decision and reason
    assert all(e["reason"] for e in d["events"] if e["type"] == "policy_decision")
    judge = [s for s in d["scores"] if s["metric"] == "judge_empathy"]
    assert judge[0]["evidence"] == {"reason": "curt"}
    assert d["categories"] == ["wrong_outcome"]


# --- Promote to regression ------------------------------------------------------------------


async def test_promote_writes_one_loadable_case_once(
    app_engine: AsyncEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scenario_mod, "REGRESSIONS_DIR", tmp_path)
    run_id = await _run(app_engine, scores=[("base/angry", 0, "outcome_match", 0.0, False)])
    async with app_engine.connect() as conn:
        first = await lab.promote(
            conn, eval_run_id=run_id, scenario_id="base/angry", run_index=0, root=tmp_path, now=NOW
        )
        again = await lab.promote(
            conn, eval_run_id=run_id, scenario_id="base/angry", run_index=0, root=tmp_path
        )
    assert first == {
        "regression_id": "regressions/base/angry",
        "file": "regressions__base__angry.yaml",
        "created": True,
    }
    assert again["created"] is False
    (case,) = load_suite(tmp_path / first["file"])
    assert case.id == "regressions/base/angry" and case.suite is EvalSuite.REGRESSIONS
    source = next(
        s for s in load_suite(scenario_mod.SUITE_FILES[EvalSuite.BASE]) if s.id == "base/angry"
    )
    assert (case.goal, case.debtor, case.expected) == (source.goal, source.debtor, source.expected)
    text = _read(tmp_path / first["file"])
    assert f"eval_run_id: {run_id}" in text and "wrong_outcome" in text
    # The regressions suite now loads it, and a failed regression maps onto the same file.
    assert [s.id for s in scenario_mod.load_suite_scenarios(EvalSuite.REGRESSIONS)] == [case.id]
    reg_run = await _run(
        app_engine,
        suite=EvalSuite.REGRESSIONS,
        scores=[("regressions/base/angry", 0, "outcome_match", 0.0, False)],
    )
    async with app_engine.connect() as conn:
        same = await lab.promote(
            conn, eval_run_id=reg_run, scenario_id="regressions/base/angry", run_index=0,
            root=tmp_path,
        )  # fmt: skip
    assert same["created"] is False and len(_listing(tmp_path)) == 1


async def test_promote_refuses_a_conversation_that_did_not_fail(
    app_engine: AsyncEngine, tmp_path: Path
) -> None:
    run_id = await _run(app_engine, scores=[("base/angry", 0, "outcome_match", 1.0, True)])
    async with app_engine.connect() as conn:
        with pytest.raises(lab.PromoteError):
            await lab.promote(
                conn, eval_run_id=run_id, scenario_id="base/angry", run_index=0, root=tmp_path
            )
    assert not _listing(tmp_path)


# --- Token estimate -------------------------------------------------------------------------


async def test_estimate_comes_from_stored_usage(app_engine: AsyncEngine) -> None:
    sid = _uid()
    judge = {"judge": {"calls": 2, "prompt_tokens": 500, "completion_tokens": 100}}
    await _run(
        app_engine,
        usage=judge,
        scores=[
            (sid, 0, "agent_tokens", 1000.0, None),
            (sid, 0, "debtor_tokens", 200.0, None),
            (sid, 0, "outcome_match", 1.0, True),
            (sid, 1, "agent_tokens", 1400.0, None),
            (sid, 1, "debtor_tokens", 200.0, None),
            (sid, 1, "outcome_match", 1.0, True),
        ],
    )
    async with app_engine.connect() as conn:
        got = await lab.token_estimate(conn, sid)
        none = await lab.token_estimate(conn, _uid())
    # mean(1200, 1600) = 1400 + judge 600 tokens / 2 conversations = 300
    assert got == {"scenario_id": sid, "n": 2, "per_side": 1700, "total": 3400}
    assert none == {"scenario_id": none["scenario_id"], "n": 0, "per_side": None, "total": None}


# --- Replay ---------------------------------------------------------------------------------


def test_replay_request_checks() -> None:
    replay.check_request("base/angry", "v1", "v3", RULES)  # every version is on rules-v8
    v6 = RULES.model_copy(update={"version": "rules-v6"})
    for args, msg in [
        (("base/angry", "v3", "v3", RULES), "must differ"),
        (("base/nope", "v1", "v3", RULES), "unknown scenario"),
        (("base/angry", "v9", "v3", RULES), "unknown agent version"),
        (("base/angry", "v1", "v3", v6), "expects rules-v8"),  # same refusal as evals.run
    ]:
        with pytest.raises(replay.ReplayError, match=msg):
            replay.check_request(*args)


async def _finish_all(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            update(Replay)
            .where(Replay.status == ReplayStatus.RUNNING)
            .values(status=ReplayStatus.FAILED, error="test cleanup")
        )


async def test_one_replay_at_a_time(app_engine: AsyncEngine) -> None:
    await _finish_all(app_engine)
    when = NOW + timedelta(days=400)
    kw: dict[str, Any] = {"scenario_id": "base/angry", "version_a": "v1", "version_b": "v3"}
    first = await replay.start(app_engine, estimate={}, running_here=True, now=when, **kw)
    with pytest.raises(replay.ReplayBusy):
        await replay.start(app_engine, estimate={}, running_here=True, now=when, **kw)
    # No live task in this process: the 'running' row was cut off by a restart.
    second = await replay.start(app_engine, estimate={}, running_here=False, now=when, **kw)
    stale = await replay.get(app_engine, first)
    assert stale is not None and (stale["status"], stale["error"]) == ("failed", replay.INTERRUPTED)
    assert second != first
    await _finish_all(app_engine)


async def test_daily_replay_limit(app_engine: AsyncEngine) -> None:
    await _finish_all(app_engine)
    when = NOW + timedelta(days=800)
    async with app_engine.begin() as conn:
        for _ in range(replay.MAX_PER_DAY):
            await conn.execute(
                insert(Replay).values(
                    scenario_id="base/angry", version_a="v1", version_b="v3",
                    status=ReplayStatus.COMPLETED, created_at=when - timedelta(hours=1),
                )
            )  # fmt: skip
    with pytest.raises(replay.ReplayLimit):
        await replay.start(
            app_engine, scenario_id="base/angry", version_a="v1", version_b="v3",
            estimate={}, running_here=False, now=when,
        )  # fmt: skip


async def test_replay_runs_both_versions_then_stores_the_comparison(
    app_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _finish_all(app_engine)
    sid = "base/angry"

    async def fake_run_suite(ctx: Any, suite: EvalSuite, k: int, only: set[str]) -> int:
        assert (suite, k, only) == (EvalSuite.BASE, 1, {sid})
        passed = ctx.agent_version == "v3"
        return await _run(
            ctx.engine, ctx.agent_version,
            scores=[(sid, 0, "outcome_match", float(passed), passed),
                    (sid, 0, "critical_violations", 0.0, True)],
        )  # fmt: skip

    monkeypatch.setattr(replay, "run_suite", fake_run_suite)
    rid = await replay.start(
        app_engine, scenario_id=sid, version_a="v1", version_b="v3",
        estimate={"total": 10}, running_here=False, now=NOW + timedelta(days=1200),
    )  # fmt: skip
    await replay.run(
        rid,
        engine=app_engine,
        rules=RULES,
        models=MODELS,
        api_key="offline",
        agent_api_key="offline",
    )
    got = await replay.get(app_engine, rid)
    assert got is not None and got["status"] == "completed", got
    assert got["run_a_id"] and got["run_b_id"] and got["label"] == "SYNTHETIC" and got["k"] == 1
    assert "v1" in got["report"] and "v3" in got["report"] and sid in got["report"]


async def test_a_failed_replay_is_stored_not_raised(
    app_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _finish_all(app_engine)

    async def broken(*_: Any) -> int:
        raise RuntimeError("groq down")

    monkeypatch.setattr(replay, "run_suite", broken)
    rid = await replay.start(
        app_engine, scenario_id="base/angry", version_a="v1", version_b="v3",
        estimate={}, running_here=False, now=NOW + timedelta(days=1600),
    )  # fmt: skip
    await replay.run(
        rid,
        engine=app_engine,
        rules=RULES,
        models=MODELS,
        api_key="offline",
        agent_api_key="offline",
    )
    got = await replay.get(app_engine, rid)
    assert got is not None and got["status"] == "failed" and "groq down" in got["error"]


# --- API ------------------------------------------------------------------------------------


async def test_lab_routes_need_the_token(api_client: httpx.AsyncClient) -> None:
    for path in ("/lab/failures", "/lab/categories", "/lab/replays"):
        res = await api_client.get(path, headers={"Authorization": "Bearer wrong"})
        assert res.status_code == 401, path


async def test_replay_api_needs_confirmation_and_a_valid_request(
    api_client: httpx.AsyncClient,
) -> None:
    body = {"scenario_id": "base/angry", "version_a": "v1", "version_b": "v3"}
    res = await api_client.post("/lab/replays", json=body)
    assert res.status_code == 400 and "confirm" in res.json()["detail"]
    # Never a valid confirmed request here: it would start a real replay (Groq tokens).
    res = await api_client.post("/lab/replays", json=body | {"confirmed": True, "version_b": "v1"})
    assert res.status_code == 400 and "must differ" in res.json()["detail"]
    assert (await api_client.get("/lab/categories")).json()[0] == "critical_violation"
    assert (await api_client.get("/lab/replays/999999999")).status_code == 404
