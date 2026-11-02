"""`events` is append-only, enforced twice:
- app role: no UPDATE/DELETE/TRUNCATE privilege (permission denied);
- trigger: rejects UPDATE/DELETE/TRUNCATE for every role, owner included.
"""

from datetime import date

import pytest
from sqlalchemy import insert, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.db.models import Debtor, Event, EventType

PERMISSION_DENIED = "permission denied"
TRIGGER_REJECTED = "events is append-only"

FORBIDDEN_ON_EVENTS = [
    pytest.param("UPDATE events SET reason = 'tampered'", id="update"),
    pytest.param("DELETE FROM events", id="delete"),
    # CASCADE: the foreign key from promises would otherwise reject a plain TRUNCATE before
    # the trigger runs, hiding whether the trigger works.
    pytest.param("TRUNCATE events CASCADE", id="truncate_cascade"),
]


async def _insert_event(conn: AsyncConnection) -> int:
    result = await conn.execute(
        insert(Event)
        .values(
            type=EventType.POLICY_DECISION,
            rule="contact_window",
            decision="blocked",
            reason="outside 08:00-21:00",
        )
        .returning(Event.id)
    )
    return result.scalar_one()


async def test_app_can_insert_and_read_events(app_engine: AsyncEngine) -> None:
    async with app_engine.connect() as conn:
        event_id = await _insert_event(conn)
        rule = await conn.scalar(text("SELECT rule FROM events WHERE id = :id"), {"id": event_id})
        await conn.rollback()
    assert rule == "contact_window"


@pytest.mark.parametrize("statement", FORBIDDEN_ON_EVENTS)
async def test_app_role_lacks_privilege(app_engine: AsyncEngine, statement: str) -> None:
    async with app_engine.connect() as conn:
        await _insert_event(conn)
        with pytest.raises(DBAPIError, match=PERMISSION_DENIED):
            await conn.execute(text(statement))
        await conn.rollback()


@pytest.mark.parametrize("statement", FORBIDDEN_ON_EVENTS)
async def test_trigger_blocks_owner(db_engine: AsyncEngine, statement: str) -> None:
    async with db_engine.connect() as conn:
        await _insert_event(conn)
        with pytest.raises(DBAPIError, match=TRIGGER_REJECTED):
            await conn.execute(text(statement))
        await conn.rollback()


async def test_app_can_update_other_tables(app_engine: AsyncEngine) -> None:
    async with app_engine.connect() as conn:
        result = await conn.execute(
            insert(Debtor)
            .values(
                external_ref="D-APP",
                full_name="App Role",
                date_of_birth=date(1990, 1, 1),
                balance_jpy=50_000,
            )
            .returning(Debtor.id)
        )
        debtor_id = result.scalar_one()
        await conn.execute(
            text("UPDATE debtors SET balance_jpy = 40000 WHERE id = :id"), {"id": debtor_id}
        )
        await conn.rollback()


async def test_app_cannot_delete_anywhere(app_engine: AsyncEngine) -> None:
    async with app_engine.connect() as conn:
        with pytest.raises(DBAPIError, match=PERMISSION_DENIED):
            await conn.execute(text("DELETE FROM debtors"))
        await conn.rollback()


async def test_app_cannot_change_migration_version(app_engine: AsyncEngine) -> None:
    async with app_engine.connect() as conn:
        with pytest.raises(DBAPIError, match=PERMISSION_DENIED):
            await conn.execute(text("UPDATE alembic_version SET version_num = 'x'"))
        await conn.rollback()
