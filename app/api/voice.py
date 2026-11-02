"""Voice calls over WebRTC: the browser offers, the server answers and runs the call."""

import asyncio
import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pydantic import BaseModel, Field

from app.agent.live import can_open
from app.api.auth import require_token
from app.db.models import Channel
from app.voice.pipeline import VoiceCall, name_keyterms

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", dependencies=[Depends(require_token)])

MAX_SDP_CHARS = 20_000


class Offer(BaseModel):
    sdp: str = Field(min_length=1, max_length=MAX_SDP_CHARS)
    type: Literal["offer"]


class Answer(BaseModel):
    sdp: str
    type: str


@router.post("/offer")
async def offer(body: Offer, conversation_id: int, request: Request) -> Answer:
    """Non-trickle ICE: the browser sends its offer once gathering is complete. The
    conversation must be a voice call that has not started (created by POST /calls)."""
    state = request.app.state
    if not await can_open(state.engine, conversation_id, Channel.VOICE):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "voice conversation not available")
    connection = SmallWebRTCConnection()
    await connection.initialize(sdp=body.sdp, type=body.type)
    call = VoiceCall(
        connection,
        conversation_id=conversation_id,
        engine=state.engine,
        rules=state.rules,
        models=state.models,
        settings=state.settings,
        keyterms=await name_keyterms(state.engine, conversation_id),
    )
    task = asyncio.create_task(call.run())
    tasks: set[asyncio.Task[Any]] = state.voice_tasks
    tasks.add(task)  # keep a reference until the call ends
    task.add_done_callback(tasks.discard)
    # Pipecat leaves get_answer() unannotated; it returns {"sdp", "type", "pc_id"}.
    answer: dict[str, Any] = connection.get_answer()  # type: ignore[no-untyped-call]
    logger.info("voice %d: answered offer", conversation_id)
    return Answer(sdp=answer["sdp"], type=answer["type"])
