"""One running chat conversation, independent of transport.

Used by /chat/ws (WebSocket) and by the eval runner (in-process), so evals exercise the
exact same agent: opening script, Flows pipeline, tools, interceptor, audit.
"""

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, cast

from pipecat.flows import FlowManager, NodeConfig
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.workers.runner import WorkerRunner
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent.actions import PAYMENT_CLAIM_SUMMARY
from app.agent.conversation import ConversationEndedError, end_conversation, open_conversation
from app.agent.pipeline import MAX_USER_CHARS, build_chat, trim_unspoken, user_frame
from app.agent.scripts import opening
from app.agent.session import ChatSession, Send
from app.agent.tools import Handler
from app.clock import OffsetClock
from app.db.audit import record_transition
from app.db.models import Channel, Conversation, Debtor, Outcome, Turn
from app.models_config import ModelsConfig
from app.policy.config import PolicyRules

logger = logging.getLogger(__name__)


async def _unstarted(engine: AsyncEngine, conversation_id: int, channel: Channel) -> Any:
    """The conversation's row if it can be opened on this channel: exists, no outcome, no turns."""
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                select(
                    Conversation.outcome,
                    Conversation.channel,
                    Conversation.agent_version,
                    Conversation.sim_started_at,
                    Conversation.debtor_id,
                    Debtor.full_name,
                )
                .join(Debtor, Debtor.id == Conversation.debtor_id)
                .where(Conversation.id == conversation_id)
            )
        ).one_or_none()
        turns = await conn.scalar(
            select(func.count()).where(Turn.conversation_id == conversation_id)
        )
    if row is None or row.outcome is not None or row.channel != channel or turns:
        return None
    return row


async def can_open(engine: AsyncEngine, conversation_id: int, channel: Channel) -> bool:
    return await _unstarted(engine, conversation_id, channel) is not None


@dataclass
class LiveChat:
    session: ChatSession
    worker: PipelineWorker
    flow: FlowManager
    handlers: dict[str, Handler]
    run_task: "asyncio.Task[Any]"
    # Pipeline errors (e.g. LLM rate limit or timeout): infrastructure, not agent behavior.
    errors: list[str] = field(default_factory=list)
    code_checks: int = 0  # identity checks run by code (v2), for tool-call ids
    context: LLMContext | None = None  # the LLM's view of the call (trimmed on barge-in)

    @classmethod
    async def start(
        cls,
        *,
        conversation_id: int,
        engine: AsyncEngine,
        rules: PolicyRules,
        models: ModelsConfig,
        api_key: str,
        send: Send,
        channel: Channel = Channel.TEXT,
    ) -> "LiveChat | None":
        """Open an unstarted conversation on its own channel (voice: guarded per sentence).
        None if it doesn't exist, is on another channel, ended, or started."""
        row = await _unstarted(engine, conversation_id, channel)
        if row is None:
            return None

        session = ChatSession(
            conversation_id=conversation_id,
            full_name=row.full_name,
            agent_version=row.agent_version,
            model_id=models.agent.model_id,
            engine=engine,
            rules=rules,
            clock=OffsetClock(row.sim_started_at),
            send=send,
        )
        async with engine.begin() as conn:
            await record_transition(
                conn,
                conversation_id=conversation_id,
                debtor_id=row.debtor_id,
                to_state="VerifyIdentity",
                reason="AI + company disclosure sent",
                at=session.clock.now(),
                agent_version=session.agent_version,
            )
        # Audit first, then speak: a hang-up right after the opening can't lose the event.
        await session.say_scripted(opening(row.full_name))

        worker, flow, node, handlers, context = build_chat(
            session,
            api_key=api_key,
            provider=models.agent.provider,
            per_sentence=channel is Channel.VOICE,
        )
        if channel is Channel.TEXT:
            # v4 retry: a system note, then the LLM writes the reply again (text only: voice
            # sentences are already spoken when a later one is blocked).
            async def _retry(feedback: str) -> None:
                note: list[Any] = [{"role": "system", "content": feedback}]
                await worker.queue_frames([LLMMessagesAppendFrame(messages=note, run_llm=True)])

            session.request_retry = _retry
            session.text_chat = True
        errors: list[str] = []

        async def _on_started(_worker: Any, _frame: Any) -> None:
            await flow.initialize(node)

        async def _on_error(_worker: Any, frame: Any) -> None:
            errors.append(str(getattr(frame, "error", frame))[:300])

        worker.add_event_handler("on_pipeline_started", _on_started)
        worker.add_event_handler("on_pipeline_error", _on_error)
        run_task = asyncio.create_task(WorkerRunner(handle_sigint=False).run(worker))
        return cls(session, worker, flow, handlers, run_task, errors, context=context)

    async def debtor_says(self, text: str) -> None:
        """Record what the debtor said and let the agent respond (unless code ended the call)."""
        text = text.strip()[:MAX_USER_CHARS]
        if not text or self.session.ended.is_set():
            return
        if self.session.wrap_outcome is not None:
            await self.session.wrap_reply(text)  # v5: scripted close, never the LLM
            return
        if await self.session.record_debtor(text):
            return  # call ended by code (identify turn cap)
        if self.session.code_should_verify():
            await self._run_by_code(text, "verify_identity", {})
            return
        if (
            await self.session.payment_claimed(text)
            and not self.session.version.fixes.payment_check
        ):
            # v3: "I already paid" goes to a dispute handoff; the LLM is not asked.
            await self._run_by_code(
                text, "request_handoff", {"reason": "dispute", "summary": PAYMENT_CLAIM_SUMMARY}
            )
            return
        await self.worker.queue_frames([user_frame(text)])

    def trim_unspoken(self, spoken_chars: int) -> None:
        """Barge-in: the LLM context keeps only what the debtor actually heard."""
        if self.context is not None:
            messages = trim_unspoken(self.context.get_messages(), spoken_chars)
            self.context.set_messages(messages)

    async def _run_by_code(self, text: str, tool: str, args: dict[str, Any]) -> None:
        """Code runs a tool the LLM would otherwise have to choose (same handler, same policy
        checks and audit): v2 verify_identity on a stated date, v3 a dispute handoff."""
        await self.worker.queue_frames([user_frame(text, run_llm=False)])
        result, next_node = await self.handlers[tool](args, self.flow)
        # D-006: the LLM context must look exactly as if the LLM had called the tool (the
        # call + its result). Without it, the negotiate node ran right after the debtor's
        # name + date of birth, and the LLM read them as a plan to propose.
        self.code_checks += 1
        call_id = f"code_{tool}_{self.code_checks}"
        call = {"id": call_id, "type": "function",
                "function": {"name": tool, "arguments": json.dumps(args)}}  # fmt: skip
        messages: list[Any] = [
            {"role": "assistant", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)},
        ]
        await self.worker.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=messages,
                    # Retry / ended: code already said the line. Need details: the LLM asks.
                    run_llm=next_node is None,
                )
            ]
        )
        if isinstance(next_node, dict):
            # Verified: the negotiate node speaks first (respond_immediately).
            await self.flow.set_node_from_config(cast(NodeConfig, next_node))

    async def close(self) -> None:
        """Stop the pipeline. A call without an outcome yet becomes `disconnected`."""
        await self.session.close_scripted()  # a v4 goodbye still pending: say the script
        await self.worker.cancel(reason="chat closed")
        results = await asyncio.gather(self.run_task, return_exceptions=True)
        if isinstance(results[0], Exception):
            logger.error("chat runner failed: %s", type(results[0]).__name__)
        s = self.session
        if s.outcome is None:
            with contextlib.suppress(ConversationEndedError):
                async with s.engine.begin() as conn:
                    convo = await open_conversation(conn, s.conversation_id)
                    await end_conversation(
                        conn,
                        convo,
                        Outcome.DISCONNECTED,
                        reason="debtor hung up or connection dropped",
                        now=s.clock.now(),
                        agent_version=s.agent_version,
                    )
            await s.end(Outcome.DISCONNECTED.value)
