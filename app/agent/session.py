"""One live chat session: turns, scripted lines, guarded LLM replies, and ending the call."""

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent.actions import payment_claimed
from app.agent.conversation import ConversationEndedError, end_conversation, open_conversation
from app.agent.output_guard import (
    AMOUNT_ASKED_AGAIN,
    SAID_ALREADY,
    asks_for_amount,
    check_agent_reply,
    drop_repeats,
    guard_agent_reply,
    retry_feedback,
    said_keys,
    sentence_key,
    strip_stock_opener,
)
from app.agent.scripts import (
    NO_VERIFICATION_END,
    SCRIPTS_VERSION,
    WARM,
    WRAP_UP_ANSWERED_CLOSE,
    WRAP_UP_CLOSE,
    WRAP_UP_QUESTION,
    WRAP_UP_SILENT_CLOSE,
    offer_line,
    plan_readback,
    without_goodbye,
)
from app.agent.versions import AgentVersion, load_version
from app.clock import Clock
from app.db.models import Conversation, Outcome, Speaker, Turn
from app.db.turns import LlmCallInfo, append_turn
from app.policy.config import PolicyRules
from app.policy.identity import find_dates

Send = Callable[[dict[str, Any]], Awaitable[None]]


# v5 wrap_up_question: a reply that asks something (about the plan just recorded).
_QUESTION = re.compile(
    r"\?|^\s*(?:when|what|how|where|which|can|could|do|does|is|will|should)", re.I
)
# v5 wrap_up_question: how long "anything else?" waits for an answer before closing.
WRAP_UP_WAIT_VOICE_S = 12.0  # ~10 s after the question has been spoken
WRAP_UP_WAIT_TEXT_S = 60.0  # typing takes longer


@dataclass
class ChatSession:
    conversation_id: int
    full_name: str
    agent_version: str
    model_id: str
    engine: AsyncEngine
    rules: PolicyRules
    clock: Clock
    send: Send
    node: str = "identify"
    outcome: str | None = None
    ended: asyncio.Event = field(default_factory=asyncio.Event)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    # What the debtor said since the last counted identity attempt. Identity is checked
    # against these words only, never against values the LLM passes to a tool.
    identity_words: list[str] = field(default_factory=list)
    # Everything the person said in this call (v3: a wrong number must be in their own words).
    debtor_words: list[str] = field(default_factory=list)
    # The agent turn being said now: voice timings and barge-in are written onto it.
    last_agent_seq: int | None = None
    # Voice: the agent turn of the LLM reply being spoken (created by its first sentence).
    reply_seq: int | None = None
    # v4 (fixes.llm_presents_terms): agent text delivered since the latest approved terms;
    # a promise needs every term said in it (Rule.TERMS_PRESENTED).
    said_since_terms: list[str] = field(default_factory=list)
    # v4: the schedule last handed to the LLM to say (an identical re-approval is not re-read).
    offered_schedule: tuple[Any, ...] = ()
    # v4 (fixes.no_repeat_offers): every plan offered in this call -> what was said while it
    # was the current plan (so a promise to an earlier plan still has its terms said).
    offered_plans: dict[tuple[Any, ...], list[str]] = field(default_factory=dict)
    # v4 (fixes.retry_blocked_reply), text chat: asks the LLM to rewrite a blocked reply.
    request_retry: Callable[[str], Awaitable[None]] | None = None
    retrying: bool = False  # the reply being written is the one rewrite after a block
    # Text chat (set by LiveChat). v4 llm_goodbyes only happen in text chats.
    text_chat: bool = False
    approach: str | None = None  # v4 (fixes.approaches): the current approach
    payment_claims_unmatched: int = 0  # v4 (fixes.payment_check)
    # v5 (V5-5): an 'unresolved' handoff is sent back once to check it is not affordability.
    unresolved_handoff_checked: bool = False
    # v5 (fixes.wrap_up_question): the recorded outcome while "anything else?" is open.
    wrap_outcome: str | None = None
    wrap_task: "asyncio.Task[None] | None" = None
    # v5: the schedule of the promise just recorded (a question about it gets a readback).
    promised_schedule: tuple[Any, ...] = ()
    # v5 (V5-6): the offer the agent already asked "what can you pay?" about.
    amount_asked_for: tuple[Any, ...] = ()
    # v5 (V5-7): every longer sentence the agent said in this call (repeat check keys).
    said_sentences: set[str] = field(default_factory=set)
    # v4 (fixes.llm_goodbyes): the outcome is recorded; the LLM's next reply is the goodbye,
    # then the call ends. closing_script is said instead if that reply is blocked or late.
    closing_outcome: str | None = None
    closing_script: str = ""
    closing_task: "asyncio.Task[None] | None" = None

    @property
    def version(self) -> AgentVersion:
        return load_version(self.agent_version)

    def code_should_verify(self) -> bool:
        """v2 (fixes.code_identity_check): the debtor's own words since the last attempt
        contain a date, so code runs the identity check instead of waiting for the LLM."""
        return (
            self.version.fixes.code_identity_check
            and self.node == "identify"
            and bool(find_dates(" ".join(self.identity_words)))
        )

    def start_closing(self, outcome: str, script: str, wait_s: float) -> None:
        """v4 goodbye: the LLM says it next; the scripted line follows if it doesn't."""
        self.closing_outcome, self.closing_script = outcome, script
        self.closing_task = asyncio.create_task(self._close_after(wait_s))

    async def _close_after(self, wait_s: float) -> None:
        await asyncio.sleep(wait_s)
        await self.close_scripted()

    async def close_scripted(self) -> None:
        """End a v4 closing with the scripted goodbye (no LLM goodbye came, or it was blocked)."""
        if self.closing_outcome is None or self.ended.is_set():
            return
        outcome = self.closing_outcome
        script = self.closing_script
        if self.wraps_up(outcome):
            script = without_goodbye(script)  # one last question follows
        await self.say_scripted(script)
        await self._finish_closing(outcome)

    async def _finish_closing(self, outcome: str) -> None:
        task, self.closing_task = self.closing_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        await self.end(outcome)

    async def say_scripted(self, text: str) -> None:
        """Fixed line rendered by code: sent verbatim, never through the LLM or interceptor.
        v4 (fixes.warm_scripts) swaps in the kinder wording of the same line."""
        if self.version.fixes.warm_scripts:
            text = WARM.get(text, text)
        self.said_since_terms.append(text)
        self.said_sentences |= said_keys(text)
        async with self.engine.begin() as conn:
            self.last_agent_seq = await append_turn(
                conn,
                conversation_id=self.conversation_id,
                speaker=Speaker.AGENT,
                text=text,
                started_at=self.clock.now(),
                llm=LlmCallInfo(model="script", prompt_version=SCRIPTS_VERSION),
            )
        await self.send({"type": "agent", "source": "script", "text": text})

    async def payment_claimed(self, text: str) -> bool:
        """v3 (fixes.payment_claim_handoff), after verification: this turn says the debt is
        already paid. Each checked turn is audited; v1/v2 never check (and read no v7 lists)."""
        if not self.version.fixes.payment_claim_handoff or self.node != "negotiate":
            return False
        async with self.engine.begin() as conn:
            return await payment_claimed(
                conn,
                conversation_id=self.conversation_id,
                debtor_text=text,
                now=self.clock.now(),
                rules=self.rules,
                agent_version=self.agent_version,
            )

    async def record_debtor(self, text: str) -> bool:
        """Record a debtor turn. Returns True if the call ended (identify turn cap reached)."""
        self.debtor_words.append(text)
        if self.node == "identify":
            self.identity_words.append(text)
        async with self.engine.begin() as conn:
            await append_turn(
                conn,
                conversation_id=self.conversation_id,
                speaker=Speaker.DEBTOR,
                text=text,
                started_at=self.clock.now(),
            )
        cap = self.rules.identity.max_turns_without_attempt
        if self.code_should_verify():
            return False  # a counted attempt follows, so the turn cap does not apply
        if self.node == "identify" and len(self.identity_words) >= cap:
            await self._end_without_verification()
            return True
        return False

    async def _end_without_verification(self) -> None:
        async with self.engine.begin() as conn:
            convo = await open_conversation(conn, self.conversation_id)
            await end_conversation(
                conn,
                convo,
                Outcome.NO_VERIFICATION,
                reason="identity not attempted within the turn limit",
                now=self.clock.now(),
                agent_version=self.agent_version,
            )
        await self.say_scripted(NO_VERIFICATION_END)
        await self.end(Outcome.NO_VERIFICATION.value)

    async def guard_and_send(
        self, raw: str, prompt_tokens: int | None, completion_tokens: int | None
    ) -> str:
        """Interceptor on an LLM reply; records the turn and sends what may be said."""
        if self.ended.is_set():
            return ""
        if self.closing_outcome is not None:
            return await self._say_goodbye(raw, prompt_tokens, completion_tokens)
        text: str | None
        v5_voice = self.version.fixes.one_option_at_a_time
        if v5_voice:  # V5-6: a stock "Thanks for letting me know." first sentence is dropped
            raw = strip_stock_opener(raw)
        retry = (
            self.version.fixes.retry_blocked_reply
            and self.request_retry is not None
            and not self.retrying
        )
        if v5_voice and self.offered_schedule and asks_for_amount(raw):
            # V5-6: one "what can you pay?" per offer; the second one goes back to the LLM
            # to call next_offer instead (text chat only: it has the rewrite step).
            if retry and self.amount_asked_for == self.offered_schedule:
                assert self.request_retry is not None
                self.retrying = True
                await self.request_retry(AMOUNT_ASKED_AGAIN)
                return ""
            self.amount_asked_for = self.offered_schedule
        if v5_voice:
            # V5-7: sentences already said in this call are dropped; a reply with nothing
            # new goes back to the LLM once (text chat), else it is sent as it was.
            fresh = drop_repeats(raw, self.said_sentences)
            if not fresh and retry:
                assert self.request_retry is not None
                self.retrying = True
                await self.request_retry(SAID_ALREADY)
                return ""
            raw = fresh or raw
        try:
            async with self.engine.begin() as conn:
                result = await check_agent_reply(
                    conn,
                    conversation_id=self.conversation_id,
                    text=raw,
                    now=self.clock.now(),
                    rules=self.rules,
                    agent_version=self.agent_version,
                )
                if retry and not result.decision.allowed:
                    # Nothing is said or stored as a turn; the block is in events.
                    self.retrying = True
                    text = None
                else:
                    self.retrying = False
                    text = result.text
                    if (
                        v5_voice
                        and not result.decision.allowed
                        and self.offered_schedule
                        and not self.said_since_terms
                    ):
                        # V5-6 (call 215): the reply presenting a new offer was blocked again;
                        # code says the offer instead of a fallback that hides it.
                        text = offer_line(list(self.offered_schedule), self.clock.now().year)
                    self.last_agent_seq = await append_turn(
                        conn,
                        conversation_id=self.conversation_id,
                        speaker=Speaker.AGENT,
                        text=text,
                        started_at=self.clock.now(),
                        llm=LlmCallInfo(
                            model=self.model_id,
                            prompt_version=f"{self.version.prompt_version}:{self.node}",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        ),
                    )
        except ConversationEndedError:
            return ""
        if text is None:
            if self.request_retry is not None:
                await self.request_retry(retry_feedback(result.decision))
            return ""
        self.said_since_terms.append(text)
        self.said_sentences |= said_keys(text)
        await self.send({"type": "agent", "source": "llm", "text": text})
        return text

    async def _say_goodbye(
        self, raw: str, prompt_tokens: int | None, completion_tokens: int | None
    ) -> str:
        """v4 closing: the LLM's goodbye, guarded like any reply; blocked -> scripted line."""
        outcome = self.closing_outcome
        assert outcome is not None
        async with self.engine.begin() as conn:
            result = await check_agent_reply(
                conn,
                conversation_id=self.conversation_id,
                text=raw,
                now=self.clock.now(),
                rules=self.rules,
                agent_version=self.agent_version,
                goodbye=True,
            )
            if result.decision.allowed:
                self.last_agent_seq = await append_turn(
                    conn,
                    conversation_id=self.conversation_id,
                    speaker=Speaker.AGENT,
                    text=result.text,
                    started_at=self.clock.now(),
                    llm=LlmCallInfo(
                        model=self.model_id,
                        prompt_version=f"{self.version.prompt_version}:goodbye",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    ),
                )
        if not result.decision.allowed:
            await self.close_scripted()
            return ""
        await self.send({"type": "agent", "source": "llm", "text": result.text})
        await self._finish_closing(outcome)
        return result.text

    async def guard_sentence(self, sentence: str) -> tuple[str, bool]:
        """Voice: interceptor on one sentence before TTS (one output_check event each). The
        reply's agent turn is created by its first sentence and grows with each one, so it
        always holds what was said. Returns (what may be said, whether it was allowed)."""
        if self.ended.is_set():
            return "", False
        if self.version.fixes.one_option_at_a_time:
            # V5-7: a sentence already said in this call is skipped (voice has no rewrite).
            key = sentence_key(sentence)
            if key and key in self.said_sentences:
                return "", True
        try:
            async with self.engine.begin() as conn:
                text = await guard_agent_reply(
                    conn,
                    conversation_id=self.conversation_id,
                    text=sentence,
                    now=self.clock.now(),
                    rules=self.rules,
                    agent_version=self.agent_version,
                )
                if self.reply_seq is None:
                    self.reply_seq = await append_turn(
                        conn,
                        conversation_id=self.conversation_id,
                        speaker=Speaker.AGENT,
                        text=text,
                        started_at=self.clock.now(),
                        llm=LlmCallInfo(
                            model=self.model_id,
                            prompt_version=f"{self.version.prompt_version}:{self.node}",
                        ),
                    )
                else:
                    await conn.execute(
                        update(Turn)
                        .where(self._turn(self.reply_seq))
                        .values(text=Turn.text + " " + text)
                    )
        except ConversationEndedError:
            return "", False
        self.last_agent_seq = self.reply_seq
        self.said_since_terms.append(text)
        self.said_sentences |= said_keys(text)
        await self.send({"type": "agent", "source": "llm", "text": text})
        return text, text == sentence.strip()

    async def end_reply(self, prompt_tokens: int | None, completion_tokens: int | None) -> None:
        """Voice: the LLM reply is complete; its token counts go on its agent turn."""
        seq, self.reply_seq = self.reply_seq, None
        if seq is None or (prompt_tokens is None and completion_tokens is None):
            return
        async with self.engine.begin() as conn:
            await conn.execute(
                update(Turn)
                .where(self._turn(seq))
                .values(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            )

    async def note_voice(self, data: dict[str, Any]) -> None:
        """Voice timings or a barge-in, merged into the latency_ms of the agent turn being
        said. Turns are not audit rows (events are), so updating one is allowed."""
        seq = self.last_agent_seq
        if seq is None:
            return
        async with self.engine.begin() as conn:
            current = await conn.scalar(select(Turn.latency_ms).where(self._turn(seq)))
            await conn.execute(
                update(Turn).where(self._turn(seq)).values(latency_ms={**(current or {}), **data})
            )

    def _turn(self, seq: int) -> Any:
        return (Turn.conversation_id == self.conversation_id) & (Turn.seq == seq)

    async def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Every LLM call, including tool-call-only replies, is added to the conversation row."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.llm_calls += 1
        async with self.engine.begin() as conn:
            await conn.execute(
                update(Conversation)
                .where(Conversation.id == self.conversation_id)
                .values(
                    llm_calls=Conversation.llm_calls + 1,
                    prompt_tokens=Conversation.prompt_tokens + prompt_tokens,
                    completion_tokens=Conversation.completion_tokens + completion_tokens,
                )
            )
        await self.send(
            {
                "type": "usage",
                "llm_calls": self.llm_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }
        )

    async def tool_result(self, name: str, result: dict[str, Any]) -> None:
        """Dev visibility: what a tool decided (shown in the dev chat page)."""
        await self.send({"type": "tool", "name": name, "result": result})

    def wraps_up(self, outcome: str) -> bool:
        """v5 (fixes.wrap_up_question): this ending gets one last "anything else?" first."""
        return (
            self.version.fixes.wrap_up_question
            and self.node == "negotiate"
            and outcome in WRAP_UP_CLOSE
        )

    async def end(self, outcome: str) -> None:
        if self.wraps_up(outcome) and self.wrap_outcome is None and not self.ended.is_set():
            # The outcome is already recorded; the call stays open for one last question.
            self.outcome = self.wrap_outcome = outcome
            await self.say_scripted(WRAP_UP_QUESTION)
            wait = WRAP_UP_WAIT_TEXT_S if self.text_chat else WRAP_UP_WAIT_VOICE_S
            self.wrap_task = asyncio.create_task(self._wrap_timeout(wait))
            return
        await self._end_now(outcome)

    async def wrap_reply(self, text: str) -> None:
        """Whatever they say after "anything else?": recorded, then a scripted close. Never
        the LLM, never a new offer (the outcome is already recorded)."""
        outcome = self.wrap_outcome
        if outcome is None or self.ended.is_set():
            return
        self._cancel_wrap_timer()
        async with self.engine.begin() as conn:
            await append_turn(
                conn,
                conversation_id=self.conversation_id,
                speaker=Speaker.DEBTOR,
                text=text,
                started_at=self.clock.now(),
            )
        close = WRAP_UP_CLOSE[outcome]
        if outcome == "promise" and self.promised_schedule and _QUESTION.search(text):
            # v5.11 text test: "when do I start?" got "You're welcome". A question about the
            # plan gets the recorded plan read back by code, then the close (no LLM, no offer).
            readback = plan_readback(list(self.promised_schedule), self.clock.now().year)
            close = f"{readback} {WRAP_UP_ANSWERED_CLOSE}"
        await self.say_scripted(close)
        await self._end_now(outcome)

    async def _wrap_timeout(self, wait_s: float) -> None:
        await asyncio.sleep(wait_s)
        if self.wrap_outcome is None or self.ended.is_set():
            return
        await self.say_scripted(WRAP_UP_SILENT_CLOSE)
        await self._end_now(self.wrap_outcome)

    def _cancel_wrap_timer(self) -> None:
        task, self.wrap_task = self.wrap_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _end_now(self, outcome: str) -> None:
        self._cancel_wrap_timer()
        self.outcome = outcome
        await self.send({"type": "end", "outcome": outcome})
        self.ended.set()
