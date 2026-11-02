"""Failure Lab replay: one scenario on agent versions A and B (k=1 each), then the comparison.

A simple in-process async task; its status lives in the replays table. At most one replay
runs at a time (a partial unique index on status = 'running'), and at most MAX_PER_DAY are
started in any 24 hours. Results are SYNTHETIC, k=1: one run per side is noise-level evidence.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent.versions import UnknownAgentVersionError, load_version
from app.db.models import Replay, ReplayStatus
from app.llm.groq_chat import GroqChat
from app.models_config import ModelsConfig
from app.policy.config import PolicyRules
from evals import compare
from evals.judge import load_rubric
from evals.report import load
from evals.run import Context, run_suite
from evals.scenario import find_scenario, suite_of

logger = logging.getLogger(__name__)

MAX_PER_DAY = 10  # deployed demo safety: every replay spends real Groq tokens
INTERRUPTED = "interrupted: the API restarted while it ran"


class ReplayError(ValueError):
    """The request is invalid (400)."""


class ReplayBusy(RuntimeError):
    """Another replay is running (409)."""


class ReplayLimit(RuntimeError):
    """The daily replay limit is reached (429)."""


def check_request(scenario_id: str, version_a: str, version_b: str, rules: PolicyRules) -> None:
    if version_a == version_b:
        raise ReplayError("versions A and B must differ")
    if find_scenario(scenario_id) is None:
        raise ReplayError(f"unknown scenario {scenario_id}")
    for v in (version_a, version_b):
        try:
            expects = load_version(v).rules_version
        except UnknownAgentVersionError as e:
            raise ReplayError(f"unknown agent version {v}") from e
        # Same refusal as evals.run: a version only runs against the rules it was built for.
        if expects != rules.version:
            raise ReplayError(f"agent {v} expects {expects}, rules.yaml is {rules.version}")


async def start(
    engine: AsyncEngine,
    *,
    scenario_id: str,
    version_a: str,
    version_b: str,
    estimate: dict[str, Any],
    running_here: bool,
    now: datetime,
) -> int:
    """Insert the replay row (status running). running_here: this process has a live replay
    task; a 'running' row without one was cut off by a restart and is marked failed."""
    async with engine.begin() as conn:
        if not running_here:
            await conn.execute(
                update(Replay)
                .where(Replay.status == ReplayStatus.RUNNING)
                .values(status=ReplayStatus.FAILED, error=INTERRUPTED, finished_at=now)
            )
        started = await conn.scalar(
            select(func.count()).where(Replay.created_at > now - timedelta(days=1))
        )
        if started and started >= MAX_PER_DAY:
            raise ReplayLimit(f"at most {MAX_PER_DAY} replays in 24 hours")
    try:
        async with engine.begin() as conn:
            replay_id = await conn.scalar(
                insert(Replay)
                .values(
                    scenario_id=scenario_id,
                    version_a=version_a,
                    version_b=version_b,
                    status=ReplayStatus.RUNNING,
                    estimate=estimate,
                    created_at=now,
                )
                .returning(Replay.id)
            )
    except IntegrityError as e:  # uq_replays_one_running
        raise ReplayBusy("another replay is running") from e
    assert replay_id is not None
    return int(replay_id)


async def run(
    replay_id: int,
    *,
    engine: AsyncEngine,
    rules: PolicyRules,
    models: ModelsConfig,
    api_key: str,
    agent_api_key: str,
) -> None:
    """The replay task: version A, then B, then compare. Never raises: failures are stored."""
    async with engine.connect() as conn:
        r = (await conn.execute(select(Replay).where(Replay.id == replay_id))).one()
    chat = GroqChat(api_key)
    values: dict[str, Any] = {}
    try:
        suite = suite_of(r.scenario_id)
        runs: dict[str, int] = {}
        for side, version in (("a", r.version_a), ("b", r.version_b)):
            ctx = Context(engine, rules, models, load_rubric(), chat, agent_api_key, version)
            runs[side] = await run_suite(ctx, suite, 1, {r.scenario_id})
            await _set(engine, replay_id, **{f"run_{side}_id": runs[side]})
        run_rows, scores = await load(engine, [runs["a"], runs["b"]])
        result = compare.build(
            [x for x in run_rows if x.id == runs["a"]],
            [x for x in run_rows if x.id == runs["b"]],
            scores,
            compare.simulated_suites(),
        )
        values = {"status": ReplayStatus.COMPLETED, "report": compare.render(result)}
    except Exception as e:  # stored on the row; the API process keeps running
        logger.error("replay %d failed: %s", replay_id, type(e).__name__)
        values = {"status": ReplayStatus.FAILED, "error": f"{type(e).__name__}: {e}"[:300]}
    finally:
        await chat.client.aclose()
        await _set(engine, replay_id, finished_at=datetime.now(UTC), **values)


async def _set(engine: AsyncEngine, replay_id: int, **values: Any) -> None:
    async with engine.begin() as conn:
        await conn.execute(update(Replay).where(Replay.id == replay_id).values(**values))


async def get(engine: AsyncEngine, replay_id: int) -> dict[str, Any] | None:
    async with engine.connect() as conn:
        r = (await conn.execute(select(Replay).where(Replay.id == replay_id))).one_or_none()
    return _as_dict(r) if r else None


async def recent(engine: AsyncEngine, limit: int = 20) -> list[dict[str, Any]]:
    async with engine.connect() as conn:
        rows = await conn.execute(select(Replay).order_by(Replay.id.desc()).limit(limit))
        return [_as_dict(r) for r in rows]


def _as_dict(r: Any) -> dict[str, Any]:
    return {
        "id": r.id,
        "scenario_id": r.scenario_id,
        "version_a": r.version_a,
        "version_b": r.version_b,
        "status": r.status.value,
        "run_a_id": r.run_a_id,
        "run_b_id": r.run_b_id,
        "estimate": r.estimate,
        "report": r.report,
        "error": r.error,
        "created_at": r.created_at.isoformat(),
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        "label": "SYNTHETIC",
        "k": 1,
    }
