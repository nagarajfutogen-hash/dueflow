"""Identity-gated account history: only current facts, with dates. DB is the source of truth."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncConnection

from app.agent.conversation import open_conversation, record
from app.agent.identity import is_verified
from app.agent.versions import load_version
from app.db.models import (
    Callback,
    Conversation,
    Debtor,
    Decision,
    Dispute,
    DisputeStatus,
    Event,
    EventType,
    Handoff,
    Payment,
    Promise,
    PromiseStatus,
)
from app.policy.engine import check_history_access


@dataclass(frozen=True)
class HistoryResult:
    allowed: bool
    facts: dict[str, Any] | None
    reason: str


async def get_account_history(
    conn: AsyncConnection, *, conversation_id: int, now: datetime, agent_version: str
) -> HistoryResult:
    convo = await open_conversation(conn, conversation_id)
    decision = check_history_access(now, await is_verified(conn, conversation_id))
    await record(conn, convo, decision, agent_version)
    if not decision.allowed:
        return HistoryResult(False, None, decision.reason)

    dispute = (
        await conn.execute(
            select(Dispute.opened_at)
            .where(Dispute.debtor_id == convo.debtor_id, Dispute.status == DisputeStatus.OPEN)
            .order_by(Dispute.opened_at.desc())
            .limit(1)
        )
    ).first()
    # Superseded promises are history, not current facts: never shown to the LLM.
    promises = (
        await conn.execute(
            select(
                Promise.status,
                Promise.kind,
                Promise.months,
                Promise.down_payment_jpy,
                Promise.installment_jpy,
                Promise.first_due_date,
                Promise.created_at,
            )
            .where(
                Promise.debtor_id == convo.debtor_id,
                Promise.status != PromiseStatus.SUPERSEDED,
            )
            .order_by(Promise.created_at.desc(), Promise.id.desc())
        )
    ).all()
    paid = (
        await conn.execute(
            select(func.coalesce(func.sum(Payment.amount_jpy), 0), func.max(Payment.paid_at)).where(
                Payment.debtor_id == convo.debtor_id
            )
        )
    ).one()

    facts: dict[str, Any] = {
        "balance_jpy": convo.balance_jpy,
        # Open dispute overrides everything: negotiation is blocked, hand off.
        "open_dispute": dispute is not None,
        "open_dispute_since": dispute.opened_at.date().isoformat() if dispute else None,
        "promises": [
            {
                "status": p.status.value,
                "kind": p.kind.value,
                "months": p.months,
                "down_payment_jpy": p.down_payment_jpy,
                "installment_jpy": p.installment_jpy,
                "first_due_date": p.first_due_date.isoformat(),
                "made_on": p.created_at.date().isoformat(),
            }
            for p in promises
        ],
        "has_broken_promise": any(p.status is PromiseStatus.BROKEN for p in promises),
        "payments_total_jpy": int(paid[0]),
        "last_payment_on": paid[1].date().isoformat() if paid[1] else None,
    }
    if load_version(agent_version).fixes.customer_context:
        facts["case_notes"] = await conn.scalar(
            select(Debtor.notes).where(Debtor.id == convo.debtor_id)
        )
        facts["previous_calls"] = await previous_calls(conn, convo.debtor_id, conversation_id)
    if load_version(agent_version).fixes.callbacks:
        # v5: this call is the callback the customer asked for (owner-approved).
        due = await conn.scalar(
            select(Callback.due_date).where(Callback.done_conversation_id == conversation_id)
        )
        if due is not None:
            facts["callback_they_asked_for"] = due.isoformat()
    return HistoryResult(True, facts, decision.reason)


PREVIOUS_CALLS_SHOWN = 3


async def previous_calls(
    conn: AsyncConnection, debtor_id: int, conversation_id: int
) -> list[dict[str, Any]]:
    """The latest finished calls before this one, as facts code knows (no transcript text,
    no figures): when, how it ended, plans offered, handoff reason, approach used."""
    rows = (
        await conn.execute(
            select(Conversation.id, Conversation.sim_started_at, Conversation.outcome)
            .where(
                Conversation.debtor_id == debtor_id,
                Conversation.id != conversation_id,
                Conversation.outcome.is_not(None),
            )
            .order_by(Conversation.id.desc())
            .limit(PREVIOUS_CALLS_SHOWN)
        )
    ).all()
    calls: list[dict[str, Any]] = []
    for cid, started, outcome in rows:
        offered = await conn.scalar(
            select(func.count())
            .select_from(Event)
            .where(
                Event.conversation_id == cid,
                Event.rule.in_(("plan_terms", "pay_in_full")),
                Event.decision == Decision.ALLOWED,
            )
        )
        handoff = await conn.scalar(select(Handoff.reason).where(Handoff.conversation_id == cid))
        approach = await conn.scalar(
            select(Event.input["approach"].astext)
            .where(
                Event.conversation_id == cid,
                Event.type == EventType.TOOL_CALL,
                Event.input["tool"].astext == "set_approach",
            )
            .order_by(Event.id.desc())
            .limit(1)
        )
        calls.append(
            {
                "date": started.date().isoformat(),
                "outcome": outcome.value if outcome else None,
                "plans_offered": int(offered or 0),
                "handoff_reason": handoff.value if handoff else None,
                "approach": approach,
            }
        )
    return calls
