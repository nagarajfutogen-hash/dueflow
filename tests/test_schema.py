import asyncio
from datetime import date
from typing import Any

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import URL, insert, inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.db.models import Base, Debtor, Dispute, Event, EventType
from tests.conftest import _recreate_database, alembic_config

EXPECTED_TABLES = {
    "debtors",
    "contact_prefs",
    "promises",
    "payments",
    "disputes",
    "conversations",
    "turns",
    "events",
}


async def _insert_debtor(conn: AsyncConnection, **overrides: Any) -> int:
    values: dict[str, Any] = {
        "external_ref": "D-TEST",
        "full_name": "Test Debtor",
        "date_of_birth": date(1990, 1, 1),
        "balance_jpy": 100_000,
    } | overrides
    result = await conn.execute(insert(Debtor).values(**values).returning(Debtor.id))
    return result.scalar_one()


async def test_all_tables_exist(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as conn:
        tables = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
    assert tables >= EXPECTED_TABLES


async def test_models_match_migrations(db_engine: AsyncEngine) -> None:
    def diff(conn: Connection) -> list[Any]:
        ctx = MigrationContext.configure(conn, opts={"compare_type": True})
        changes: list[Any] = compare_metadata(ctx, Base.metadata)
        return changes

    async with db_engine.connect() as conn:
        assert await conn.run_sync(diff) == []


async def test_debtor_defaults_to_tokyo(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as conn:
        debtor_id = await _insert_debtor(conn)
        tz = await conn.scalar(
            text("SELECT timezone FROM debtors WHERE id = :id"), {"id": debtor_id}
        )
        await conn.rollback()
    assert tz == "Asia/Tokyo"


async def test_negative_balance_rejected(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as conn:
        with pytest.raises(IntegrityError, match="balance_non_negative"):
            await _insert_debtor(conn, balance_jpy=-1)
        await conn.rollback()


async def test_policy_decision_event_requires_rule_decision_reason(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as conn:
        with pytest.raises(IntegrityError, match="policy_decision_complete"):
            await conn.execute(insert(Event).values(type=EventType.POLICY_DECISION))
        await conn.rollback()


async def test_non_policy_event_needs_no_decision(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as conn:
        await conn.execute(insert(Event).values(type=EventType.ERROR, reason="timeout"))
        await conn.rollback()


async def test_unknown_enum_value_rejected(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as conn:
        with pytest.raises(IntegrityError, match="eventtype"):
            await conn.execute(text("INSERT INTO events (type) VALUES ('made_up')"))
        await conn.rollback()


async def test_resolved_dispute_needs_resolved_at(db_engine: AsyncEngine) -> None:
    async with db_engine.connect() as conn:
        debtor_id = await _insert_debtor(conn)
        with pytest.raises(IntegrityError, match="resolved_at_matches_status"):
            await conn.execute(
                insert(Dispute).values(
                    debtor_id=debtor_id,
                    status="resolved",
                    reason="paid already",
                    opened_at=text("now()"),
                )
            )
        await conn.rollback()


def test_downgrade_and_upgrade_roundtrip(test_db_url: URL) -> None:
    # Own empty database: the shared test DB holds rows (e.g. newer outcome values) that an
    # older CHECK constraint would rightly reject on downgrade.
    url = test_db_url.set(database=f"{test_db_url.database}_roundtrip")
    asyncio.run(_recreate_database(url))
    cfg = alembic_config(url)
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
