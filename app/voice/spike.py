"""Day 5 spike (dev only): browser mic -> WebRTC -> Deepgram STT -> echo -> Deepgram TTS.

No LLM, no database. It answers one question before the real voice pipeline is built: does
WebRTC media reach the API container (Docker Desktop locally, then the VPS)? Logs carry
timings and lengths only, never what was said.
"""

import asyncio
import logging
import time
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner
from pydantic import BaseModel, Field

from app.api.auth import require_token
from app.config import Settings
from app.models_config import ModelsConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice/spike", dependencies=[Depends(require_token)])

ECHO_PREFIX = "I heard"
MAX_SDP_CHARS = 20_000


class Offer(BaseModel):
    sdp: str = Field(min_length=1, max_length=MAX_SDP_CHARS)
    type: Literal["offer"]


class Answer(BaseModel):
    sdp: str
    type: str


class Echo(FrameProcessor):
    """Final transcripts come back as speech; times transcript -> first bot audio."""

    def __init__(self) -> None:
        super().__init__()
        self._heard_at: float | None = None
        self.latencies_ms: list[float] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._heard_at = time.monotonic()
            logger.info("spike: final transcript (%d chars)", len(frame.text))
            await self.push_frame(TTSSpeakFrame(f"{ECHO_PREFIX}: {frame.text}"), direction)
            return
        if isinstance(frame, BotStartedSpeakingFrame) and self._heard_at is not None:
            ms = (time.monotonic() - self._heard_at) * 1000
            self.latencies_ms.append(ms)
            self._heard_at = None
            logger.info("spike: transcript -> first bot audio %.0f ms", ms)
        await self.push_frame(frame, direction)


def build_worker(
    connection: SmallWebRTCConnection, settings: Settings, models: ModelsConfig
) -> PipelineWorker:
    transport = SmallWebRTCTransport(
        connection, TransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )
    stt = DeepgramSTTService(
        api_key=settings.deepgram_api_key,
        settings=DeepgramSTTService.Settings(model=models.stt.model_id),
    )
    tts = DeepgramTTSService(api_key=settings.deepgram_api_key, voice=models.tts.model_id)
    pipeline = Pipeline([transport.input(), stt, Echo(), tts, transport.output()])
    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True))

    async def _gone(_transport: Any, _client: Any) -> None:
        logger.info("spike: client disconnected")
        await worker.cancel(reason="client disconnected")

    transport.add_event_handler("on_client_disconnected", _gone)
    return worker


@router.post("/offer")
async def offer(body: Offer, request: Request) -> Answer:
    """Non-trickle ICE: the browser sends its offer once gathering is complete."""
    settings: Settings = request.app.state.settings
    connection = SmallWebRTCConnection()
    await connection.initialize(sdp=body.sdp, type=body.type)
    worker = build_worker(connection, settings, request.app.state.models)
    task = asyncio.create_task(WorkerRunner(handle_sigint=False).run(worker))
    tasks: set[asyncio.Task[Any]] = request.app.state.voice_tasks
    tasks.add(task)  # keep a reference until the call ends
    task.add_done_callback(tasks.discard)
    # Pipecat leaves get_answer() unannotated; it returns {"sdp", "type", "pc_id"}.
    answer: dict[str, Any] = connection.get_answer()  # type: ignore[no-untyped-call]
    logger.info("spike: answered offer pc_id=%s", answer.get("pc_id"))
    return Answer(sdp=answer["sdp"], type=answer["type"])
