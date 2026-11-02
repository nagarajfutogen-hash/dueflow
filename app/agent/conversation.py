"""Shared helpers for an in-progress conversation: load it, and end it with an outcome."""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from app.db.audit import record_decision, record_transition
from app.db.models import Conversation, Debtor, Outcome
from app.policy.engine import Decision


class ConversationEndedError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenConversation:
    conversation_id: int
    debtor_id: int
    full_name: str
    timezone: str
    balance_jpy: int


async def open_conversation(
    conn: AsyncConnection, conversation_id: int, *, allow_ended: bool = False
) -> OpenConversation:
    """Load a conversation that has not ended yet; raise if it has (unless allow_ended:
    the v4 goodbye line, checked after the outcome is recorded)."""
    row = (
        await conn.execute(
            select(
                Conversation.outcome,
                Debtor.id,
                Debtor.full_name,
                Debtor.timezone,
                Debtor.balance_jpy,
            )
            .join(Debtor, Debtor.id == Conversation.debtor_id)
            .where(Conversation.id == conversation_id)
        )
    ).one()
    if not allow_ended and row.outcome is not None:
        raise ConversationEndedError(f"conversation {conversation_id} already ended")
    return OpenConversation(conversation_id, row.id, row.full_name, row.timezone, row.balance_jpy)


async def record(
    conn: AsyncConnection, convo: OpenConversation, decision: Decision, agent_version: str
) -> int:
    """Record a policy decision for this conversation; returns the event id."""
    return await record_decision(
        conn,
        decision,
        agent_version=agent_version,
        conversation_id=convo.conversation_id,
        debtor_id=convo.debtor_id,
    )


async def end_conversation(
    conn: AsyncConnection,
    convo: OpenConversation,
    outcome: Outcome,
    *,
    reason: str,
    now: datetime,
    agent_version: str,
) -> None:
    await conn.execute(
        update(Conversation)
        .where(Conversation.id == convo.conversation_id)
        .values(outcome=outcome, ended_at=now)
    )
    await record_transition(
        conn,
        conversation_id=convo.conversation_id,
        debtor_id=convo.debtor_id,
        to_state="End",
        reason=reason,
        at=now,
        agent_version=agent_version,
        detail={"outcome": outcome.value},
    )
