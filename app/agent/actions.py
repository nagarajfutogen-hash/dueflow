"""Call-ending and preference actions: stop-contact, no-workplace, handoff."""

from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import exists, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection

from app.agent import scripts
from app.agent.conversation import end_conversation, open_conversation, record
from app.agent.identity import is_verified
from app.db.audit import record_tool_call
from app.db.models import (
    ContactPrefs,
    Dispute,
    DisputeStatus,
    Handoff,
    HandoffReason,
    Outcome,
    Payment,
)
from app.policy.config import PolicyRules
from app.policy.engine import (
    Decision,
    PaymentClaim,
    PaymentOnFile,
    check_payment_claim,
    check_payment_record,
)

SUMMARY_MAX_CHARS = 1000


async def record_stop_contact(
    conn: AsyncConnection, *, conversation_id: int, now: datetime, agent_version: str
) -> str:
    """Honored even before verification: when in doubt, stop. The flag is never cleared."""
    convo = await open_conversation(conn, conversation_id)
    stmt = pg_insert(ContactPrefs).values(
        debtor_id=convo.debtor_id, stop_contact=True, stop_contact_at=now
    )
    await conn.execute(
        stmt.on_conflict_do_update(
            index_elements=[ContactPrefs.debtor_id],
            set_={"stop_contact": True, "updated_at": now},
            where=ContactPrefs.stop_contact.is_(False),
        )
    )
    verified = await is_verified(conn, conversation_id)
    await record_tool_call(
        conn,
        conversation_id=conversation_id,
        debtor_id=convo.debtor_id,
        tool="record_stop_contact",
        at=now,
        agent_version=agent_version,
        detail={"identity_verified": verified},
    )
    await end_conversation(
        conn,
        convo,
        Outcome.STOP_CONTACT,
        reason="debtor requested stop contact",
        now=now,
        agent_version=agent_version,
    )
    return scripts.STOP_CONTACT_END


async def record_no_workplace_contact(
    conn: AsyncConnection, *, conversation_id: int, now: datetime, agent_version: str
) -> str:
    """'Don't call my work' is a permanent flag. The call continues."""
    convo = await open_conversation(conn, conversation_id)
    stmt = pg_insert(ContactPrefs).values(debtor_id=convo.debtor_id, no_workplace_contact=True)
    await conn.execute(
        stmt.on_conflict_do_update(
            index_elements=[ContactPrefs.debtor_id],
            set_={"no_workplace_contact": True, "updated_at": now},
        )
    )
    await record_tool_call(
        conn,
        conversation_id=conversation_id,
        debtor_id=convo.debtor_id,
        tool="record_no_workplace_contact",
        at=now,
        agent_version=agent_version,
    )
    return scripts.NO_WORKPLACE_ACK


async def request_handoff(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    reason: HandoffReason,
    summary: str,
    now: datetime,
    agent_version: str,
) -> str:
    """Hand off to a human. A newly raised dispute is also stored, which blocks negotiation
    on every future call until a human resolves it."""
    convo = await open_conversation(conn, conversation_id)
    summary = summary.strip()[:SUMMARY_MAX_CHARS]
    if reason is HandoffReason.DISPUTE:
        already_open = await conn.scalar(
            select(
                exists().where(
                    Dispute.debtor_id == convo.debtor_id, Dispute.status == DisputeStatus.OPEN
                )
            )
        )
        if not already_open:
            await conn.execute(
                insert(Dispute).values(
                    debtor_id=convo.debtor_id,
                    status=DisputeStatus.OPEN,
                    reason=summary,
                    opened_at=now,
                )
            )
    await conn.execute(
        insert(Handoff).values(
            conversation_id=conversation_id,
            debtor_id=convo.debtor_id,
            reason=reason,
            summary=summary,
        )
    )
    await end_conversation(
        conn,
        convo,
        Outcome.HANDOFF,
        reason=f"handoff: {reason.value}",
        now=now,
        agent_version=agent_version,
    )
    return scripts.HANDOFF_END


PAYMENT_CLAIM_SUMMARY = "Debtor says the balance is already paid (payment claim detected by code)."


async def payment_claimed(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    debtor_text: str,
    now: datetime,
    rules: PolicyRules,
    agent_version: str,
) -> bool:
    """v3, after verification: True if the person's turn claims the debt is already paid.
    Every checked turn writes a payment_claim decision."""
    convo = await open_conversation(conn, conversation_id)
    decision = check_payment_claim(now, debtor_text=debtor_text, rules=rules)
    await record(conn, convo, decision, agent_version)
    return not decision.allowed


async def payments_on_file(conn: AsyncConnection, debtor_id: int, tz: str) -> list[PaymentOnFile]:
    rows = await conn.execute(
        select(Payment.amount_jpy, Payment.paid_at, Payment.method, Payment.reference)
        .where(Payment.debtor_id == debtor_id)
        .order_by(Payment.paid_at, Payment.id)
    )
    zone = ZoneInfo(tz)
    return [
        PaymentOnFile(r.amount_jpy, r.paid_at.astimezone(zone).date(), r.method, r.reference)
        for r in rows
    ]


async def check_payment_claim_record(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    claim: PaymentClaim,
    now: datetime,
    agent_version: str,
) -> tuple[Decision, int]:
    """v4: check what the person says they paid against payments on record (audited).
    Returns the decision and the current balance."""
    convo = await open_conversation(conn, conversation_id)
    payments = await payments_on_file(conn, convo.debtor_id, convo.timezone)
    decision = check_payment_record(now, claim=claim, payments=payments)
    await record(conn, convo, decision, agent_version)
    return decision, convo.balance_jpy
