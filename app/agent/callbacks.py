"""Agent v5 (V5-3): a callback the customer asked for. Code turns their words into a date.

The LLM passes the customer's own words for the day ("Friday", "the 10th"); code resolves
them against today (debtor-local) and against everything the customer said in this call,
never trusting a date the LLM worked out. The callback is saved as 'requested': nobody is
called until the owner approves it on the admin dashboard.
"""

from dataclasses import dataclass, replace
from datetime import date, datetime
from zoneinfo import ZoneInfo

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncConnection

from app.agent import scripts
from app.agent.conversation import end_conversation, open_conversation, record
from app.agent.identity import is_verified
from app.db.models import Callback, CallbackStatus, Outcome, Speaker, Turn
from app.policy.callback import check_callback, customer_dates
from app.policy.config import PolicyRules
from app.policy.engine import check_identity_required

REASONS = ("payday", "needs_time", "busy_now", "other")


@dataclass(frozen=True)
class CallbackResult:
    recorded: bool
    due: date | None = None
    codes: tuple[str, ...] = ()
    say: str = ""  # the scripted goodbye, when recorded


async def record_callback(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    day_phrase: str,
    reason: str,
    now: datetime,
    rules: PolicyRules,
    agent_version: str,
) -> CallbackResult:
    convo = await open_conversation(conn, conversation_id)
    identity = check_identity_required(
        now, identity_verified=await is_verified(conn, conversation_id), action="record_callback"
    )
    await record(conn, convo, identity, agent_version)
    if not identity.allowed:
        return CallbackResult(False, codes=identity.codes)
    said = list(
        await conn.scalars(
            select(Turn.text).where(
                Turn.conversation_id == conversation_id, Turn.speaker == Speaker.DEBTOR
            )
        )
    )
    today = now.astimezone(ZoneInfo(convo.timezone)).date()
    # The earliest future date the phrase names; the check below confirms the customer said it.
    future = sorted(d for d in customer_dates([day_phrase], today) if d > today)
    requested = future[0] if future else today
    decision = check_callback(
        now, debtor_tz=convo.timezone, requested=requested, said=said, rules=rules
    )
    decision = replace(decision, input=decision.input | {"day_phrase": day_phrase[:60]})
    await record(conn, convo, decision, agent_version)
    if not decision.allowed:
        return CallbackResult(False, codes=decision.codes)
    await conn.execute(
        insert(Callback).values(
            debtor_id=convo.debtor_id,
            conversation_id=conversation_id,
            due_date=requested,
            reason=reason if reason in REASONS else "other",
            customer_words=str(decision.input.get("customer_phrase") or day_phrase)[:200],
            status=CallbackStatus.REQUESTED,
        )
    )
    await end_conversation(
        conn,
        convo,
        Outcome.CALLBACK_SCHEDULED,
        reason="customer asked for a callback",
        now=now,
        agent_version=agent_version,
    )
    return CallbackResult(True, due=requested, say=scripts.callback_end(requested))
