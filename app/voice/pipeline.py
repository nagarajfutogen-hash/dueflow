"""Voice call: browser WebRTC -> Silero VAD -> Deepgram STT -> the same agent -> Deepgram TTS.

The agent is the text LiveChat unchanged (same Flows, tools, policy checks and audit), opened
on a voice conversation, so its replies are guarded per sentence. This module only moves
words: a user turn's final transcripts become one debtor line, and every line the agent may
say (guarded LLM sentences, scripted lines word for word) is spoken by TTS. Logs carry ids,
timings and lengths only, never what was said.
"""

import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    DataFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent.live import LiveChat
from app.config import Settings
from app.db.models import Channel, Conversation, Debtor
from app.models_config import ModelsConfig
from app.policy.config import PolicyRules
from app.voice.latency import turn_latency

logger = logging.getLogger(__name__)

# Pipecat defaults. The turn ends on VAD silence + a timeout, never Smart Turn (dropped).
VAD_PARAMS = VADParams()
USER_SPEECH_TIMEOUT_S = 0.6

OnText = Callable[[str], Awaitable[None]]
OnData = Callable[[dict[str, Any]], Awaitable[None]]
Now = Callable[[], float]

_YEN = re.compile(r"(?:¥|\bJPY)\s?(\d[\d,]*)")


def for_speech(text: str) -> str:
    """What TTS reads aloud: "¥12,000" / "JPY 12,000" become "12,000 yen". Voice only, after
    the guard: turns and events keep the guarded text exactly as checked."""
    return _YEN.sub(r"\1 yen", text)


@dataclass
class _TurnEnded(DataFrame):
    """Queued behind the turn's transcripts: the turn-stop frame is a system frame and is
    handled ahead of data frames still queued, so it can overtake its last transcript."""


class VoiceBridge(FrameProcessor):
    """The final transcripts of one user turn become one debtor line for the agent."""

    def __init__(self, on_turn: OnText) -> None:
        super().__init__()
        self._on_turn = on_turn
        self._heard: list[str] = []
        self._waiting = False  # the turn ended before any transcript arrived

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):  # finals only (interims are another class)
            if text := frame.text.strip():
                self._heard.append(text)
            if self._waiting and self._heard:
                await self._flush()
            return  # transcripts stop here: TTS must never speak the debtor's words
        if isinstance(frame, _TurnEnded):
            if self._heard:
                await self._flush()
            else:
                self._waiting = True
            return
        if isinstance(frame, UserStartedSpeakingFrame):
            self._waiting = False
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self.queue_frame(_TurnEnded(), direction)
        await self.push_frame(frame, direction)

    async def _flush(self) -> None:
        text = " ".join(self._heard)
        self._heard.clear()
        self._waiting = False
        await self._on_turn(text)


class VoiceTimer(FrameProcessor):
    """Between TTS and the output: the first TTS audio after VAD said the user stopped."""

    def __init__(self, on_latency: OnData, vad_stop_secs: float, now: Now = time.monotonic):
        super().__init__()
        self._on_latency = on_latency
        self._vad_stop_secs = vad_stop_secs
        self._now = now
        self._vad_end: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._vad_end = None  # still talking: only the last stop counts
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_end = self._now()
        elif isinstance(frame, TTSAudioRawFrame) and self._vad_end is not None:
            data = turn_latency(self._vad_end, self._now(), self._vad_stop_secs)
            self._vad_end = None
            await self._on_latency(data)
        await self.push_frame(frame, direction)


class SpokenTracker(FrameProcessor):
    """After the output: what the debtor heard since they last spoke. A debtor who speaks
    over the agent (interruption while the bot is speaking) is reported as a barge-in."""

    def __init__(self, on_barge_in: OnData, now: Now = time.monotonic) -> None:
        super().__init__()
        self._on_barge_in = on_barge_in
        self._now = now
        self._spoken: list[str] = []
        self._speaking_since: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._speaking_since = self._now()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._speaking_since = None
        elif isinstance(frame, TTSTextFrame):
            self._spoken.append(frame.text.strip())
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._spoken.clear()
        elif isinstance(frame, InterruptionFrame) and self._speaking_since is not None:
            spoken = " ".join(t for t in self._spoken if t)
            at_ms = round((self._now() - self._speaking_since) * 1000)
            self._spoken.clear()
            self._speaking_since = None
            await self._on_barge_in(
                {"interrupted": True, "interrupted_at_ms": at_ms, "spoken_chars": len(spoken)}
            )
        await self.push_frame(frame, direction)


class VoiceProbe(FrameProcessor):
    """Debug: counts of frame types passing one point of the pipeline. Logs the first of each
    type and a summary when the call ends; counts and type names only, never what was said."""

    WATCH = (
        "InputAudioRawFrame",
        "VADUserStartedSpeakingFrame",
        "VADUserStoppedSpeakingFrame",
        "InterimTranscriptionFrame",
        "TranscriptionFrame",
        "UserStartedSpeakingFrame",
        "UserStoppedSpeakingFrame",
    )

    def __init__(self, where: str) -> None:
        super().__init__()
        self._where = where
        self._counts: dict[str, int] = {}

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        name = type(frame).__name__
        if name in self.WATCH:
            self._counts[name] = self._counts.get(name, 0) + 1
            if self._counts[name] == 1:
                logger.info("voice probe %s: first %s", self._where, name)
        elif name in ("EndFrame", "CancelFrame"):
            logger.info("voice probe %s: counts %s", self._where, self._counts)
        await self.push_frame(frame, direction)


def build_voice_worker(
    transport: SmallWebRTCTransport,
    settings: Settings,
    models: ModelsConfig,
    *,
    on_turn: OnText,
    on_latency: OnData,
    on_barge_in: OnData,
    keyterms: list[str] | None = None,
) -> PipelineWorker:
    stt = DeepgramSTTService(
        api_key=settings.deepgram_api_key,
        # smart_format writes spoken dates as "August 8, 1993", which identity can read.
        # keyterms (V5-7): the debtor's name words, so nova-3 spells them right (live call 3
        # heard a first name as "data"; identity needs every name word).
        settings=DeepgramSTTService.Settings(
            model=models.stt.model_id, smart_format=True, keyterm=keyterms or []
        ),
    )
    turns = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=USER_SPEECH_TIMEOUT_S)]
        )
    )
    tts = DeepgramTTSService(api_key=settings.deepgram_api_key, voice=models.tts.model_id)
    pipeline = Pipeline(
        [
            transport.input(),
            VoiceProbe("input"),
            VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VAD_PARAMS)),
            VoiceProbe("vad"),
            stt,
            VoiceProbe("stt"),
            turns,
            VoiceProbe("turns"),
            VoiceBridge(on_turn),
            tts,
            VoiceTimer(on_latency, VAD_PARAMS.stop_secs),
            transport.output(),
            SpokenTracker(on_barge_in),
        ]
    )
    return PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True))


async def name_keyterms(engine: AsyncEngine, conversation_id: int) -> list[str]:
    """The debtor's name words for STT keyterm prompting (never logged)."""
    async with engine.connect() as conn:
        name = await conn.scalar(
            select(Debtor.full_name)
            .join(Conversation, Conversation.debtor_id == Debtor.id)
            .where(Conversation.id == conversation_id)
        )
    return [w for w in (name or "").split() if w]


class VoiceCall:
    """One voice conversation: the transport pipeline plus the agent (opened on connect, so
    the debtor hears the opening line)."""

    def __init__(
        self,
        connection: SmallWebRTCConnection,
        *,
        conversation_id: int,
        engine: AsyncEngine,
        rules: PolicyRules,
        models: ModelsConfig,
        settings: Settings,
        keyterms: list[str] | None = None,
    ) -> None:
        self.conversation_id = conversation_id
        self._engine, self._rules, self._models, self._settings = engine, rules, models, settings
        self.live: LiveChat | None = None
        transport = SmallWebRTCTransport(
            connection, TransportParams(audio_in_enabled=True, audio_out_enabled=True)
        )
        self.worker = build_voice_worker(
            transport,
            settings,
            models,
            on_turn=self._debtor_said,
            on_latency=self._latency,
            on_barge_in=self._barge_in,
            keyterms=keyterms,
        )
        transport.add_event_handler("on_client_connected", self._connected)
        transport.add_event_handler("on_client_disconnected", self._disconnected)

    async def run(self) -> None:
        try:
            await WorkerRunner(handle_sigint=False).run(self.worker)
        finally:
            if self.live is not None:
                await self.live.close()  # no outcome yet: the call becomes `disconnected`
            logger.info("voice %d: call closed", self.conversation_id)

    async def send(self, message: dict[str, Any]) -> None:
        """Everything the agent may say is spoken (amounts read as yen); the end of the call
        ends the pipeline after the last line has been spoken."""
        if message.get("type") == "agent" and message.get("text"):
            await self.worker.queue_frames([TTSSpeakFrame(for_speech(message["text"]))])
        elif message.get("type") == "end":
            await self.worker.queue_frames([EndFrame()])

    async def _connected(self, _transport: Any, _client: Any) -> None:
        self.live = await LiveChat.start(
            conversation_id=self.conversation_id,
            engine=self._engine,
            rules=self._rules,
            models=self._models,
            api_key=self._settings.agent_api_key(self._models.agent.provider),
            send=self.send,
            channel=Channel.VOICE,
        )
        if self.live is None:
            logger.info("voice %d: conversation not available", self.conversation_id)
            await self.worker.cancel(reason="conversation not available")

    async def _disconnected(self, _transport: Any, _client: Any) -> None:
        await self.worker.cancel(reason="client disconnected")

    async def _debtor_said(self, text: str) -> None:
        if self.live is not None:
            logger.info("voice %d: debtor turn (%d chars)", self.conversation_id, len(text))
            await self.live.debtor_says(text)

    async def _latency(self, data: dict[str, Any]) -> None:
        if self.live is not None:
            await self.live.session.note_voice(data)

    async def _barge_in(self, data: dict[str, Any]) -> None:
        if self.live is not None:
            await self.live.session.note_voice(data)
            self.live.trim_unspoken(int(data["spoken_chars"]))
