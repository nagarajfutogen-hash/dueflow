"""Failure Lab API: failures list, conversation detail, promote to regression, replay A vs B.

Read-only except promote (writes one YAML file to evals/regressions/) and replay (spends Groq
tokens; token-protected like every route here, one at a time, capped per day).
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.auth import require_token
from app.db.models import ResultLabel
from evals import lab, replay

router = APIRouter(prefix="/lab", dependencies=[Depends(require_token)])


@router.get("/failures")
async def failures(
    request: Request,
    category: str | None = None,
    version: str | None = None,
    label: ResultLabel | None = None,
    history: bool = False,
) -> list[dict[str, Any]]:
    async with request.app.state.engine.connect() as conn:
        return await lab.failures(
            conn,
            category=category,
            version=version,
            label=label.value if label else None,
            history=history,
        )


@router.get("/categories")
async def categories() -> list[str]:
    return list(lab.CATEGORIES.values())


@router.get("/conversations/{conversation_id}")
async def conversation(conversation_id: int, request: Request) -> dict[str, Any]:
    async with request.app.state.engine.connect() as conn:
        detail = await lab.conversation_detail(conn, conversation_id)
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    return detail


class PromoteIn(BaseModel):
    eval_run_id: int
    scenario_id: str = Field(min_length=1, max_length=128)
    run_index: int = Field(ge=0)


@router.post("/regressions")
async def promote(body: PromoteIn, request: Request) -> dict[str, Any]:
    async with request.app.state.engine.connect() as conn:
        try:
            return await lab.promote(
                conn,
                eval_run_id=body.eval_run_id,
                scenario_id=body.scenario_id,
                run_index=body.run_index,
            )
        except lab.PromoteError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e


@router.get("/replays/estimate")
async def estimate(scenario_id: str, request: Request) -> dict[str, Any]:
    async with request.app.state.engine.connect() as conn:
        return await lab.token_estimate(conn, scenario_id)


class ReplayIn(BaseModel):
    scenario_id: str = Field(min_length=1, max_length=128)
    version_a: str = Field(min_length=1, max_length=64)
    version_b: str = Field(min_length=1, max_length=64)
    confirmed: bool = False  # the owner saw the token estimate and confirmed


@router.post("/replays", status_code=status.HTTP_202_ACCEPTED)
async def start_replay(body: ReplayIn, request: Request) -> dict[str, Any]:
    state = request.app.state
    if not body.confirmed:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "confirm the token estimate first")
    try:
        replay.check_request(body.scenario_id, body.version_a, body.version_b, state.rules)
    except replay.ReplayError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e
    task: asyncio.Task[None] | None = getattr(state, "replay_task", None)
    async with state.engine.connect() as conn:
        tokens = await lab.token_estimate(conn, body.scenario_id)
    try:
        replay_id = await replay.start(
            state.engine,
            scenario_id=body.scenario_id,
            version_a=body.version_a,
            version_b=body.version_b,
            estimate=tokens,
            running_here=task is not None and not task.done(),
            now=datetime.now(UTC),
        )
    except replay.ReplayBusy as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e
    except replay.ReplayLimit as e:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(e)) from e
    state.replay_task = asyncio.create_task(
        replay.run(
            replay_id,
            engine=state.engine,
            rules=state.rules,
            models=state.models,
            api_key=state.settings.groq_api_key,
            agent_api_key=state.settings.agent_api_key(state.models.agent.provider),
        )
    )
    return {"id": replay_id, "status": "running", "estimate": tokens}


@router.get("/replays")
async def replays(request: Request) -> list[dict[str, Any]]:
    return await replay.recent(request.app.state.engine)


@router.get("/replays/{replay_id}")
async def get_replay(replay_id: int, request: Request) -> dict[str, Any]:
    found = await replay.get(request.app.state.engine, replay_id)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "replay not found")
    return found
