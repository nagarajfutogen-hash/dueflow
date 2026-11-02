"""Chat pipeline: user aggregator -> LLM -> guarded sink -> assistant aggregator.

The guarded sink holds back streamed LLM text, runs the output interceptor on the whole
reply, then pushes only the guarded text downstream. So the assistant aggregator stores
what was actually said, not what the LLM first wrote.

Voice (per_sentence): each sentence is checked as soon as it is complete and sent to TTS.
Sentences are split exactly where the output check splits a reply, and a blocked sentence
ends the reply early (fallback said, rest dropped), so voice says what text would send.
"""

from typing import Any, Protocol

from pipecat.flows import FlowManager, NodeConfig
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    MetricsFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.openai.llm import OpenAILLMService

from app.agent.nodes import identify_node
from app.agent.scripts import opening
from app.agent.session import ChatSession
from app.agent.tools import Handler, make_handlers, to_flows_node
from app.agent.versions import AgentVersion
from app.policy.output_check import SENTENCE_SPLIT

LLM_RETRY_TIMEOUT_S = 20.0
MAX_USER_CHARS = 2000


class Guard(Protocol):
    """What the sink needs from a session (lets tests use a stub)."""

    async def guard_and_send(
        self, raw: str, prompt_tokens: int | None, completion_tokens: int | None
    ) -> str: ...

    async def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None: ...

    async def guard_sentence(self, sentence: str) -> tuple[str, bool]: ...

    async def end_reply(self, prompt_tokens: int | None, completion_tokens: int | None) -> None: ...


def take_sentences(buffer: str) -> tuple[list[str], str]:
    """Complete sentences at the start of streamed text, and the unfinished rest. A sentence
    is complete once whitespace follows its end mark (the output check's split)."""
    parts = SENTENCE_SPLIT.split(buffer.lstrip())
    return [p for p in parts[:-1] if p], parts[-1]


class GuardedTextSink(FrameProcessor):
    def __init__(self, session: Guard, *, per_sentence: bool = False) -> None:
        super().__init__()
        self._session = session
        self._per_sentence = per_sentence
        self._parts: list[str] = []
        self._usage: tuple[int | None, int | None] = (None, None)
        self._said: list[str] = []  # per_sentence: what was sent to TTS in this reply
        self._blocked = False  # per_sentence: a blocked sentence ended this reply

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._parts.clear()
            self._said.clear()
            self._blocked = False
        elif isinstance(frame, LLMTextFrame):
            self._parts.append(frame.text)
            if self._per_sentence:
                done, rest = take_sentences("".join(self._parts))
                self._parts = [rest]
                for sentence in done:
                    await self._speak(sentence)
            return  # held back until the interceptor has seen it
        elif isinstance(frame, MetricsFrame):
            for item in frame.data:
                if isinstance(item, LLMUsageMetricsData):
                    usage = item.value
                    self._usage = (usage.prompt_tokens, usage.completion_tokens)
                    # Counted for every LLM call, including tool-call-only replies.
                    await self._session.add_usage(usage.prompt_tokens, usage.completion_tokens)
        elif isinstance(frame, LLMFullResponseEndFrame) and self._per_sentence:
            rest = "".join(self._parts).strip()
            self._parts.clear()
            if rest:
                await self._speak(rest)
            await self._session.end_reply(*self._usage)
            self._usage = (None, None)
            if self._said:
                await self.push_frame(LLMTextFrame(text=" ".join(self._said)), direction)
        elif isinstance(frame, LLMFullResponseEndFrame):
            raw = "".join(self._parts).strip()
            self._parts.clear()
            if raw:
                text = await self._session.guard_and_send(raw, *self._usage)
                self._usage = (None, None)
                if text:
                    await self.push_frame(LLMTextFrame(text=text), direction)
        await self.push_frame(frame, direction)

    async def _speak(self, sentence: str) -> None:
        if self._blocked:
            return  # the reply already ended at a blocked sentence
        text, allowed = await self._session.guard_sentence(sentence)
        if text:
            self._said.append(text)
        self._blocked = not allowed


def trim_unspoken(messages: list[Any], spoken_chars: int) -> list[Any]:
    """Barge-in: the agent's lines since the debtor last spoke, cut to what was played. Tool
    calls and results stay; a plain line nobody heard is removed."""
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    left = spoken_chars
    out = list(messages[: last_user + 1])
    for m in messages[last_user + 1 :]:
        content = m.get("content")
        if m.get("role") != "assistant" or not isinstance(content, str) or m.get("tool_calls"):
            out.append(m)
            continue
        if left >= len(content):
            out.append(m)
        elif left > 0:
            out.append({**m, "content": content[:left].rstrip() + " —"})
        left = max(0, left - len(content) - 1)  # -1: the space between joined lines
    return out


def make_llm(
    provider: str, api_key: str, model_id: str, version: AgentVersion
) -> GroqLLMService | OpenAILLMService:
    """The agent LLM for config/models.yaml agent.provider (both use the OpenAI-style API)."""
    if provider == "openai":
        # gpt-4.1 models take no reasoning_effort; the version's other settings apply.
        return OpenAILLMService(
            api_key=api_key,
            settings=OpenAILLMService.Settings(model=model_id, temperature=version.temperature),
            retry_timeout_secs=LLM_RETRY_TIMEOUT_S,
            retry_on_timeout=True,
        )
    return GroqLLMService(
        api_key=api_key,
        settings=GroqLLMService.Settings(
            model=model_id,
            temperature=version.temperature,
            reasoning_effort=version.reasoning_effort,
        ),
        retry_timeout_secs=LLM_RETRY_TIMEOUT_S,
        retry_on_timeout=True,
    )


def build_chat(
    session: ChatSession,
    *,
    api_key: str,
    provider: str = "groq",
    per_sentence: bool = False,
) -> tuple[PipelineWorker, FlowManager, NodeConfig, dict[str, Handler], LLMContext]:
    version = session.version
    llm = make_llm(provider, api_key, session.model_id, version)
    # The fixed opening was already said by code; the LLM sees it as its own first line.
    context = LLMContext(messages=[{"role": "assistant", "content": opening(session.full_name)}])
    aggregators = LLMContextAggregatorPair(context)
    pipeline = Pipeline(
        [
            aggregators.user(),
            llm,
            GuardedTextSink(session, per_sentence=per_sentence),
            aggregators.assistant(),
        ]
    )
    worker = PipelineWorker(
        pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True)
    )
    # Pipecat's LLMService generic is invariant in its adapter type; GroqLLMService is valid here.
    flow = FlowManager(worker=worker, llm=llm, context_aggregator=aggregators)  # type: ignore[arg-type]
    handlers = make_handlers(session)
    node = to_flows_node(identify_node(session.full_name, version), handlers)
    return worker, flow, node, handlers, context


def user_frame(text: str, *, run_llm: bool = True) -> LLMMessagesAppendFrame:
    return LLMMessagesAppendFrame(
        messages=[{"role": "user", "content": text[:MAX_USER_CHARS]}], run_llm=run_llm
    )
