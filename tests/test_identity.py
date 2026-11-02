import re
from datetime import UTC, date, datetime

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent import scripts
from app.agent.conversation import ConversationEndedError
from app.agent.identity import (
    ThirdPartyKind,
    VerificationResult,
    VerificationStatus,
    handle_third_party,
    is_verified,
    verify_identity,
)
from app.db.models import Conversation, Event, Outcome
from app.policy.config import load_rules
from tests.conftest import MakeDebtor

RULES = load_rules()
NOW = datetime(2026, 10, 24, 3, tzinfo=UTC)
NAME = "Hanako Verifytest"
DOB = date(1980, 2, 29)
DEBT_WORDS = re.compile(
    r"debt|owe|balance|payment|pay|loan|collect|overdue|amount|account|¥|yen|\d", re.I
)


async def _conversation(client: httpx.AsyncClient, make_debtor: MakeDebtor) -> int:
    ref = await make_debtor(full_name=NAME, dob=DOB)
    body = {
        "debtor_ref": ref,
        "sim_time": "2026-10-24T12:00:00+09:00",
        "target": "mobile",
        "agent_version": "v1",
    }
    conversation_id = (await client.post("/calls", json=body)).json()["conversation_id"]
    assert conversation_id is not None
    return int(conversation_id)


async def _verify(
    engine: AsyncEngine, conversation_id: int, name: str, dob: str
) -> VerificationResult:
    async with engine.begin() as conn:
        return await verify_identity(
            conn,
            conversation_id=conversation_id,
            debtor_text=f"My name is {name}, born {dob}.",
            now=NOW,
            rules=RULES,
            agent_version="v1",
        )


async def _outcome(engine: AsyncEngine, conversation_id: int) -> Outcome | None:
    async with engine.connect() as conn:
        return await conn.scalar(
            select(Conversation.outcome).where(Conversation.id == conversation_id)
        )


async def test_correct_details_verify(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _conversation(api_client, make_debtor)
    result = await _verify(app_engine, cid, "verifytest hanako", "February 29, 1980")
    assert result.status is VerificationStatus.VERIFIED
    assert result.say == scripts.VERIFIED
    async with app_engine.connect() as conn:
        assert await is_verified(conn, cid) is True


async def test_verify_again_after_success_adds_no_event(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _conversation(api_client, make_debtor)
    await _verify(app_engine, cid, NAME, "1980-02-29")
    again = await _verify(app_engine, cid, "wrong", "1970-01-01")
    assert again.status is VerificationStatus.VERIFIED
    async with app_engine.connect() as conn:
        rules = (
            await conn.execute(select(Event.rule).where(Event.conversation_id == cid))
        ).scalars()
        assert list(rules).count("identity_verification") == 1


async def test_one_failure_retries_with_same_message_for_any_field(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    wrong_name = await _verify(
        app_engine, await _conversation(api_client, make_debtor), "Someone Else", "1980-02-29"
    )
    wrong_dob = await _verify(
        app_engine, await _conversation(api_client, make_debtor), NAME, "1980-03-01"
    )
    for result in (wrong_name, wrong_dob):
        assert result.status is VerificationStatus.RETRY
        assert result.attempts_left == 1
    assert wrong_name.say == wrong_dob.say == scripts.VERIFY_RETRY


async def test_two_failures_end_as_wrong_person(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _conversation(api_client, make_debtor)
    await _verify(app_engine, cid, "Nobody", "1990-01-01")
    final = await _verify(app_engine, cid, "Nobody", "1990-01-01")
    assert final.status is VerificationStatus.FAILED_FINAL
    assert final.say == scripts.VERIFY_FAILED_END
    assert await _outcome(app_engine, cid) is Outcome.END_WRONG_PERSON
    with pytest.raises(ConversationEndedError):
        await _verify(app_engine, cid, NAME, "1980-02-29")


async def test_third_party_not_available_is_callback_not_failed_attempt(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _conversation(api_client, make_debtor)
    async with app_engine.begin() as conn:
        say = await handle_third_party(
            conn,
            conversation_id=cid,
            kind=ThirdPartyKind.NOT_AVAILABLE,
            now=NOW,
            agent_version="v1",
        )
    assert NAME in say and scripts.NEUTRAL_COMPANY_NAME in say
    assert "Servicing" not in say
    assert await _outcome(app_engine, cid) is Outcome.THIRD_PARTY_CALLBACK
    async with app_engine.connect() as conn:
        rules = list(
            (await conn.execute(select(Event.rule).where(Event.conversation_id == cid))).scalars()
        )
    assert "identity_verification" not in rules


async def test_wrong_number_ends_as_wrong_person(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _conversation(api_client, make_debtor)
    async with app_engine.begin() as conn:
        say = await handle_third_party(
            conn,
            conversation_id=cid,
            kind=ThirdPartyKind.WRONG_NUMBER,
            now=NOW,
            agent_version="v1",
        )
    assert say == scripts.WRONG_NUMBER_END
    assert await _outcome(app_engine, cid) is Outcome.END_WRONG_PERSON


async def test_identity_events_hold_no_pii(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _conversation(api_client, make_debtor)
    await _verify(app_engine, cid, "Guessed Person", "1975-06-06")
    async with app_engine.connect() as conn:
        rows = (
            await conn.execute(
                select(Event.input, Event.reason).where(
                    Event.conversation_id == cid, Event.rule == "identity_verification"
                )
            )
        ).all()
    logged = str(rows)
    for pii in ("Hanako", "Verifytest", "1980", "Guessed", "1975"):
        assert pii not in logged


def test_pre_verification_scripts_never_mention_the_debt() -> None:
    lines = [
        scripts.opening(NAME),
        scripts.third_party_callback(NAME),
        *scripts.PRE_VERIFICATION_LINES,
    ]
    for line in lines:
        # The callback number is the only allowed digits; strip it before checking.
        checked = line.replace(scripts.CALLBACK_NUMBER, "")
        assert not DEBT_WORDS.search(checked), line


def test_opening_discloses_ai_and_neutral_company_only() -> None:
    line = scripts.opening(NAME)
    assert "AI" in line and scripts.NEUTRAL_COMPANY_NAME in line
    assert "Servicing" not in line


def test_pre_verification_lines_never_name_the_full_company() -> None:
    lines = [scripts.opening(NAME), scripts.third_party_callback(NAME)]
    for line in [*lines, *scripts.PRE_VERIFICATION_LINES]:
        assert "Servicing" not in line, line


def test_full_company_name_given_once_verified() -> None:
    assert scripts.COMPANY_NAME in scripts.VERIFIED


async def test_no_date_stated_is_not_counted(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _conversation(api_client, make_debtor)
    async with app_engine.begin() as conn:
        first = await verify_identity(
            conn,
            conversation_id=cid,
            debtor_text=f"Yes, this is {NAME}.",
            now=NOW,
            rules=RULES,
            agent_version="v1",
        )
    assert first.status is VerificationStatus.NEED_DETAILS
    # Both real attempts are still available.
    assert (await _verify(app_engine, cid, "Nobody", "1990-01-01")).status is (
        VerificationStatus.RETRY
    )
    final = await _verify(app_engine, cid, NAME, "1980-02-29")
    assert final.status is VerificationStatus.VERIFIED
