from datetime import UTC, datetime

import pytest
from sqlalchemy import insert, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.db.models import (
    Channel,
    Conversation,
    Debtor,
    Handoff,
    HandoffReason,
    PhoneType,
    ResultLabel,
    Speaker,
)
from app.db.seed import SEED_DEBTORS, seed
from app.db.turns import LlmCallInfo, append_turn
from tests.conftest import MakeDebtor

NOW = datetime(2026, 10, 24, 3, tzinfo=UTC)


async def _conversation(conn: AsyncConnection, ref: str) -> tuple[int, int]:
    debtor_id = await conn.scalar(select(Debtor.id).where(Debtor.external_ref == ref))
    assert debtor_id is not None
    conversation_id = (
        await conn.execute(
            insert(Conversation)
            .values(
                debtor_id=debtor_id,
                channel=Channel.TEXT,
                label=ResultLabel.SYNTHETIC,
                agent_version="v1",
                sim_started_at=NOW,
                dialed_phone_type=PhoneType.MOBILE,
                account_snapshot={},
            )
            .returning(Conversation.id)
        )
    ).scalar_one()
    return debtor_id, conversation_id


async def test_seed_is_idempotent_and_covers_personas(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as conn:
        first = await seed(conn)
        second = await seed(conn)
        rows = await conn.execute(select(Debtor.external_ref, Debtor.timezone))
        tz_by_ref = {ref: tz for ref, tz in rows}
        await conn.rollback()
    assert first == [d.ref for d in SEED_DEBTORS]
    assert second == []
    assert any(tz != "Asia/Tokyo" for ref, tz in tz_by_ref.items() if ref.startswith("D-"))
    personas = {d.persona for d in SEED_DEBTORS}
    assert {"returning_broken_promise", "returning_open_dispute", "returning_stop_contact"} <= (
        personas
    )


async def test_append_turn_increments_seq(app_engine: AsyncEngine, make_debtor: MakeDebtor) -> None:
    ref = await make_debtor()
    async with app_engine.connect() as conn:
        _, conversation_id = await _conversation(conn, ref)
        first = await append_turn(
            conn,
            conversation_id=conversation_id,
            speaker=Speaker.AGENT,
            text="hello",
            started_at=NOW,
            llm=LlmCallInfo(model="m", prompt_version="p1", prompt_tokens=10),
        )
        second = await append_turn(
            conn, conversation_id=conversation_id, speaker=Speaker.DEBTOR, text="hi", started_at=NOW
        )
        await conn.rollback()
    assert (first, second) == (1, 2)


async def test_app_can_record_handoff(app_engine: AsyncEngine, make_debtor: MakeDebtor) -> None:
    ref = await make_debtor()
    async with app_engine.connect() as conn:
        debtor_id, conversation_id = await _conversation(conn, ref)
        await conn.execute(
            insert(Handoff).values(
                conversation_id=conversation_id,
                debtor_id=debtor_id,
                reason=HandoffReason.HARDSHIP,
                summary="lost job, asks for a call back",
            )
        )
        with pytest.raises(DBAPIError, match="permission denied"):
            await conn.execute(text("DELETE FROM handoffs"))
        await conn.rollback()


async def test_unknown_outcome_rejected(db_engine: AsyncEngine, make_debtor: MakeDebtor) -> None:
    ref = await make_debtor()
    async with db_engine.connect() as conn:
        _, conversation_id = await _conversation(conn, ref)
        with pytest.raises(IntegrityError, match="ck_conversations_outcome"):
            await conn.execute(
                text("UPDATE conversations SET outcome = 'made_up' WHERE id = :id"),
                {"id": conversation_id},
            )
        await conn.rollback()
