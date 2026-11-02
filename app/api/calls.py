from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import AfterValidator, BaseModel, Field

from app.agent.versions import load_version
from app.api.auth import require_token
from app.calls import (
    DebtorNotFoundError,
    DialRequest,
    DialResult,
    NoNumberForTargetError,
    auto_dial,
    dial,
)
from app.clock import SystemClock
from app.db.models import Channel, ResultLabel
from app.policy.engine import DialTarget

router = APIRouter(dependencies=[Depends(require_token)])


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("sim_time must include a timezone offset")
    return value


def _known_version(value: str) -> str:
    load_version(value)  # ValueError (-> 422) when agent_versions/<value>.yaml is missing
    return value


class CallIn(BaseModel):
    debtor_ref: str = Field(min_length=1, max_length=32)
    sim_time: Annotated[datetime, AfterValidator(_aware)]
    target: DialTarget
    agent_version: Annotated[
        str, Field(min_length=1, max_length=64), AfterValidator(_known_version)
    ]
    label: ResultLabel = ResultLabel.SYNTHETIC
    channel: Channel = Channel.TEXT
    scenario_id: str | None = Field(default=None, max_length=128)
    seed: int | None = None


class DecisionOut(BaseModel):
    rule: str
    allowed: bool
    reason: str
    codes: list[str]


class CallOut(BaseModel):
    allowed: bool
    conversation_id: int | None
    decisions: list[DecisionOut]
    event_ids: list[int]


@router.post("/calls")
async def create_call(body: CallIn, request: Request) -> CallOut:
    """Simulated outbound dial. Blocked dials return 200 with allowed=false (a valid outcome)."""
    req = DialRequest(**body.model_dump())
    try:
        async with request.app.state.engine.begin() as conn:
            result = await dial(conn, req, request.app.state.rules)
    except DebtorNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown debtor") from exc
    except NoNumberForTargetError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return _call_out(result)


def _call_out(result: DialResult) -> CallOut:
    return CallOut(
        allowed=result.allowed,
        conversation_id=result.conversation_id,
        decisions=[
            DecisionOut(rule=d.rule.value, allowed=d.allowed, reason=d.reason, codes=list(d.codes))
            for d in result.decisions
        ],
        event_ids=result.event_ids,
    )


class AutoCallIn(BaseModel):
    # None: the server picks the debtor too. Either way it picks the time and number.
    debtor_ref: str | None = Field(default=None, min_length=1, max_length=32)
    agent_version: Annotated[
        str, Field(min_length=1, max_length=64), AfterValidator(_known_version)
    ]
    label: ResultLabel = ResultLabel.HUMAN_TEST
    channel: Channel = Channel.TEXT


class AutoCallOut(CallOut):
    debtor_ref: str
    sim_time: datetime
    target: DialTarget


@router.post("/calls/auto")
async def create_auto_call(body: AutoCallIn, request: Request) -> AutoCallOut:
    """Dev page: the server picks the debtor (least recently dialed), a time inside their
    window and a number. The normal pre-call check still decides and is recorded."""
    try:
        async with request.app.state.engine.begin() as conn:
            picked = await auto_dial(
                conn,
                request.app.state.rules,
                today=SystemClock().now().date(),
                agent_version=body.agent_version,
                label=body.label,
                channel=body.channel,
                debtor_ref=body.debtor_ref,
            )
    except DebtorNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown debtor") from exc
    if picked is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "no debtor can be called right now")
    pick, result = picked
    return AutoCallOut(
        **_call_out(result).model_dump(),
        debtor_ref=pick.debtor_ref,
        sim_time=pick.sim_time,
        target=pick.target,
    )
