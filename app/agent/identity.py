"""Identity verification and third-party handling for one conversation.

The LLM only extracts what the person said and calls these; code decides. Results given
back to the LLM say verified / retry / ended, never which field failed.
"""

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncConnection

from app.agent import scripts
from app.agent.conversation import (
    end_conversation,
    open_conversation,
    record,
)
from app.db.models import Debtor, Decision, Event, Outcome
from app.policy.config import PolicyRules
from app.policy.engine import Rule, check_identity, check_wrong_number_claim


class VerificationStatus(enum.StrEnum):
    VERIFIED = "verified"
    NEED_DETAILS = "need_details"  # no date of birth stated yet: not an attempt
    RETRY = "retry"
    FAILED_FINAL = "failed_final"


class ThirdPartyKind(enum.StrEnum):
    NOT_AVAILABLE = "not_available"  # "he's not here" -> callback request
    WRONG_NUMBER = "wrong_number"  # "no such person here"


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    say: str
    attempts_left: int


def _identity_events(conversation_id: int) -> Any:
    return select(func.count()).where(
        Event.conversation_id == conversation_id,
        Event.rule == Rule.IDENTITY_VERIFICATION.value,
    )


async def is_verified(conn: AsyncConnection, conversation_id: int) -> bool:
    """True once an allowed identity_verification event exists in this conversation."""
    count = await conn.scalar(
        _identity_events(conversation_id).where(Event.decision == Decision.ALLOWED)
    )
    return bool(count)


async def failed_attempts(conn: AsyncConnection, conversation_id: int) -> int:
    """Counted attempts only: a wrong date of birth or name. 'no_details' is not an attempt."""
    count = await conn.scalar(
        _identity_events(conversation_id).where(
            Event.input["codes"].contains(["identity_mismatch"])
        )
    )
    return count or 0


async def verify_identity(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    debtor_text: str,
    now: datetime,
    rules: PolicyRules,
    agent_version: str,
) -> VerificationResult:
    convo = await open_conversation(conn, conversation_id)
    max_failed = rules.identity.max_failed_attempts
    if await is_verified(conn, conversation_id):
        return VerificationResult(VerificationStatus.VERIFIED, scripts.VERIFIED, max_failed)

    failed = await failed_attempts(conn, conversation_id)
    actual_dob = await conn.scalar(select(Debtor.date_of_birth).where(Debtor.id == convo.debtor_id))
    assert actual_dob is not None
    decision = check_identity(
        now,
        debtor_text=debtor_text,
        actual_name=convo.full_name,
        actual_dob=actual_dob,
        failed_attempts=failed,
        rules=rules,
    )
    await record(conn, convo, decision, agent_version)
    if decision.allowed:
        return VerificationResult(
            VerificationStatus.VERIFIED, scripts.VERIFIED, max_failed - failed
        )
    if decision.codes == ("no_details",):
        return VerificationResult(VerificationStatus.NEED_DETAILS, "", max_failed - failed)
    attempts_left = max(0, max_failed - failed - 1)
    if "max_attempts_reached" in decision.codes:
        await end_conversation(
            conn,
            convo,
            Outcome.END_WRONG_PERSON,
            reason="identity verification failed",
            now=now,
            agent_version=agent_version,
        )
        return VerificationResult(VerificationStatus.FAILED_FINAL, scripts.VERIFY_FAILED_END, 0)
    return VerificationResult(VerificationStatus.RETRY, scripts.VERIFY_RETRY, attempts_left)


async def confirm_wrong_number(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    debtor_text: str,
    now: datetime,
    rules: PolicyRules,
    agent_version: str,
) -> bool:
    """v3: the LLM may only end the call as a wrong number if the person said so. Audited."""
    convo = await open_conversation(conn, conversation_id)
    decision = check_wrong_number_claim(
        now, debtor_text=debtor_text, debtor_name=convo.full_name, rules=rules
    )
    await record(conn, convo, decision, agent_version)
    return decision.allowed


async def handle_third_party(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    kind: ThirdPartyKind,
    now: datetime,
    agent_version: str,
) -> str:
    """Someone other than the debtor answered. Not a failed verification attempt."""
    convo = await open_conversation(conn, conversation_id)
    if kind is ThirdPartyKind.NOT_AVAILABLE:
        outcome, say = Outcome.THIRD_PARTY_CALLBACK, scripts.third_party_callback(convo.full_name)
    else:
        outcome, say = Outcome.END_WRONG_PERSON, scripts.WRONG_NUMBER_END
    await end_conversation(
        conn,
        convo,
        outcome,
        reason=f"third party: {kind.value}",
        now=now,
        agent_version=agent_version,
    )
    return say
