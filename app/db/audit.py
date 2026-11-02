"""Writes policy decisions to the append-only `events` table.

Kept outside app/policy/ so the policy engine stays free of database code.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from app.db.models import Decision as DecisionValue
from app.db.models import Event, EventType
from app.policy.engine import Decision


async def record_decision(
    conn: AsyncConnection,
    decision: Decision,
    *,
    agent_version: str,
    conversation_id: int | None = None,
    debtor_id: int | None = None,
) -> int:
    """Insert one policy_decision event (allowed or blocked) and return its id."""
    result = await conn.execute(
        insert(Event)
        .values(
            type=EventType.POLICY_DECISION,
            conversation_id=conversation_id,
            debtor_id=debtor_id,
            rule=decision.rule.value,
            decision=DecisionValue.ALLOWED if decision.allowed else DecisionValue.BLOCKED,
            reason=decision.reason,
            input=decision.input | {"codes": list(decision.codes)},
            agent_version=agent_version,
            clock_time=decision.at,
        )
        .returning(Event.id)
    )
    return result.scalar_one()


async def record_decisions(
    conn: AsyncConnection,
    decisions: list[Decision],
    *,
    agent_version: str,
    conversation_id: int | None = None,
    debtor_id: int | None = None,
) -> list[int]:
    return [
        await record_decision(
            conn,
            d,
            agent_version=agent_version,
            conversation_id=conversation_id,
            debtor_id=debtor_id,
        )
        for d in decisions
    ]


async def record_transition(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    debtor_id: int,
    to_state: str,
    reason: str,
    at: datetime,
    agent_version: str,
    detail: dict[str, Any] | None = None,
) -> int:
    """Insert one state_transition event (e.g. VerifyIdentity -> End) and return its id."""
    result = await conn.execute(
        insert(Event)
        .values(
            type=EventType.STATE_TRANSITION,
            conversation_id=conversation_id,
            debtor_id=debtor_id,
            reason=reason,
            input={"to": to_state} | (detail or {}),
            agent_version=agent_version,
            clock_time=at,
        )
        .returning(Event.id)
    )
    return result.scalar_one()


async def record_tool_call(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    debtor_id: int,
    tool: str,
    at: datetime,
    agent_version: str,
    detail: dict[str, Any] | None = None,
) -> int:
    """Insert one tool_call event for a side effect that is not a policy decision."""
    result = await conn.execute(
        insert(Event)
        .values(
            type=EventType.TOOL_CALL,
            conversation_id=conversation_id,
            debtor_id=debtor_id,
            reason=tool,
            input={"tool": tool} | (detail or {}),
            agent_version=agent_version,
            clock_time=at,
        )
        .returning(Event.id)
    )
    return result.scalar_one()
