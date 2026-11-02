"""Runs one scenario end to end against the real agent, in-process.

Creates a fresh SYNTHETIC debtor, dials through the real pre-call check, then drives
LiveChat (the same code /chat/ws uses) with scripted lines or the simulator.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent.live import LiveChat
from app.calls import DialRequest, dial
from app.db.models import Conversation, ResultLabel
from app.db.seed import SeedDebtor, SeedPromise, insert_debtor
from app.llm.groq_chat import GroqChat, Usage
from app.models_config import ModelsConfig
from app.policy.config import PolicyRules
from evals.scenario import BLOCKED_PRECALL, Kind, Scenario
from evals.simulate import Turn, next_line

AGENT_REPLY_TIMEOUT_S = 75.0  # first agent message after a debtor line (LLM + tools)
AGENT_IDLE_S = 3.5  # then: quiet this long = the agent has finished its turn


@dataclass
class ConversationResult:
    scenario_id: str
    run_index: int
    conversation_id: int | None
    outcome: str | None
    transcript: list[Turn] = field(default_factory=list)
    ended_by: str = ""  # agent | debtor_hangup | script_done | turn_limit | agent_silent
    infra_errors: list[str] = field(default_factory=list)
    debtor_usage: Usage = field(default_factory=Usage)
    debtor_raw: list[str] = field(default_factory=list)  # simulator outputs (evidence)
    dropped_hangups: list[int] = field(default_factory=list)  # debtor_raw indexes (sim-v3)


def seed_debtor(s: Scenario, ref: str) -> SeedDebtor:
    d = s.debtor
    return SeedDebtor(
        ref=ref,
        persona=s.id,
        full_name=d.full_name,
        dob=d.dob,
        balance_jpy=d.balance_jpy,
        timezone=d.timezone,
        mobile=d.mobile,
        home=d.home,
        work=d.work,
        stop_contact=d.stop_contact,
        no_workplace_contact=d.no_workplace_contact,
        workplace_justification=d.workplace_justification,
        open_dispute=d.open_dispute,
        promises=tuple(
            SeedPromise(p.months, p.down_payment_jpy, p.installment_jpy, p.first_due_date, p.status)
            for p in d.promises
        ),
        payments_jpy=d.payments_jpy,
    )


async def agent_turn(outbox: "asyncio.Queue[dict[str, Any]]") -> list[dict[str, Any]]:
    """Collect the agent's messages for one turn: wait for the first, then until quiet."""
    messages: list[dict[str, Any]] = []
    timeout = AGENT_REPLY_TIMEOUT_S
    while True:
        try:
            msg = await asyncio.wait_for(outbox.get(), timeout)
        except TimeoutError:
            return messages
        messages.append(msg)
        if msg["type"] == "end":
            return messages
        if msg["type"] in ("agent", "end"):
            timeout = AGENT_IDLE_S


def _agent_text(messages: list[dict[str, Any]]) -> str:
    return " ".join(m["text"] for m in messages if m["type"] == "agent")


async def run_scenario(
    s: Scenario,
    run_index: int,
    *,
    engine: AsyncEngine,
    rules: PolicyRules,
    models: ModelsConfig,
    chat: GroqChat,
    agent_api_key: str,
    agent_version: str,
    seed: int,
) -> ConversationResult:
    ref = f"EV-{uuid.uuid4().hex[:10]}"
    async with engine.begin() as conn:
        await insert_debtor(conn, seed_debtor(s, ref))
        dialed = await dial(
            conn,
            DialRequest(
                debtor_ref=ref,
                sim_time=s.dial.sim_time,
                target=s.dial.target,
                agent_version=agent_version,
                label=ResultLabel.SYNTHETIC,
                scenario_id=s.id,
                seed=seed,
            ),
            rules,
        )
    if not dialed.allowed:
        return ConversationResult(
            s.id, run_index, None, BLOCKED_PRECALL, ended_by="blocked_precall"
        )
    assert dialed.conversation_id is not None
    cid = dialed.conversation_id

    outbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    live = await LiveChat.start(
        conversation_id=cid,
        engine=engine,
        rules=rules,
        models=models,
        api_key=agent_api_key,
        send=outbox.put,
    )
    assert live is not None
    result = ConversationResult(s.id, run_index, cid, None)
    try:
        opening = await agent_turn(outbox)
        result.transcript.append(Turn("agent", _agent_text(opening)))
        script = [s.line(raw) for raw in s.script]
        for turn_no in range(s.max_turns):
            if live.session.ended.is_set():
                result.ended_by = "agent"
                break
            hang_up = False
            if s.kind is Kind.SCRIPTED:
                if turn_no >= len(script):
                    result.ended_by = "script_done"
                    break
                line: str | None = script[turn_no]
            else:
                said, usage = await next_line(
                    chat,
                    models.simulator.model_id,
                    s,
                    result.transcript,
                    seed + turn_no,
                    allow_drop=not result.dropped_hangups,
                )
                for u in usage:
                    result.debtor_usage.add(u)
                if said.hang_up_dropped:
                    result.dropped_hangups.append(len(result.debtor_raw))
                result.debtor_raw.append(said.raw)
                line, hang_up = said.text, said.hang_up
                if line is None:
                    result.ended_by = "debtor_hangup"
                    break
            assert line is not None  # both branches above guarantee it
            result.transcript.append(Turn("debtor", line))
            await live.debtor_says(line)
            reply = await agent_turn(outbox)
            if live.errors:
                result.infra_errors = list(live.errors)
                result.ended_by = "infra_error"
                break
            text = _agent_text(reply)
            if text:
                result.transcript.append(Turn("agent", text))
            elif not live.session.ended.is_set():
                result.ended_by = "agent_silent"
                break
            if hang_up and not live.session.ended.is_set():
                result.ended_by = "debtor_hangup"  # after the agent answered them
                break
        else:
            result.ended_by = "turn_limit"
        if live.session.ended.is_set() and not result.ended_by:
            result.ended_by = "agent"
    finally:
        await live.close()

    async with engine.connect() as conn:
        outcome = await conn.scalar(select(Conversation.outcome).where(Conversation.id == cid))
    result.outcome = outcome.value if outcome else None
    return result
