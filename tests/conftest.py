import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import replace
from datetime import date
from typing import Any

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import URL, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import Settings, load_settings
from app.db.seed import SeedDebtor, insert_debtor
from app.db.session import make_engine
from app.main import app
from app.models_config import load_models
from app.policy.config import load_rules


def _test_settings() -> Settings:
    settings = load_settings()
    name = os.environ.get("POSTGRES_TEST_DB", f"{settings.postgres_db}_test")
    return replace(settings, postgres_db=name)


async def _recreate_database(url: URL) -> None:
    # Connect to the maintenance DB to drop/create the test DB.
    admin = create_async_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f'DROP DATABASE IF EXISTS "{url.database}" WITH (FORCE)'))
        await conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    await admin.dispose()


def alembic_config(url: URL) -> Config:
    cfg = Config("alembic.ini")
    cfg.attributes["database_url"] = url
    return cfg


@pytest.fixture(scope="session")
def test_settings() -> Iterator[Settings]:
    """Fresh test database migrated to head, once per test session."""
    settings = _test_settings()
    asyncio.run(_recreate_database(settings.owner_url))
    command.upgrade(alembic_config(settings.owner_url), "head")
    yield settings


@pytest.fixture
def test_db_url(test_settings: Settings) -> URL:
    return test_settings.owner_url


@pytest.fixture
async def db_engine(test_settings: Settings) -> AsyncIterator[AsyncEngine]:
    """Connected as the schema owner."""
    engine = make_engine(test_settings.owner_url)
    yield engine
    await engine.dispose()


@pytest.fixture
async def app_engine(test_settings: Settings) -> AsyncIterator[AsyncEngine]:
    """Connected as the restricted app role, like the API."""
    engine = make_engine(test_settings.app_url)
    yield engine
    await engine.dispose()


TEST_API_TOKEN = "test-token"


@pytest.fixture
async def api_client(
    test_settings: Settings, app_engine: AsyncEngine
) -> AsyncIterator[httpx.AsyncClient]:
    """API client against the test database, authenticated with TEST_API_TOKEN."""
    app.state.settings = replace(test_settings, api_token=TEST_API_TOKEN)
    app.state.engine = app_engine
    app.state.rules = load_rules()
    app.state.models = load_models()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {TEST_API_TOKEN}"},
    ) as client:
        yield client


MakeDebtor = Callable[..., Awaitable[str]]


@pytest.fixture
def make_debtor(db_engine: AsyncEngine) -> MakeDebtor:
    """Insert a fresh debtor with a unique ref (committed). Conversations and events can't
    be deleted, so tests never share a debtor: attempt limits would leak between tests."""

    async def _make(**overrides: Any) -> str:
        fields: dict[str, Any] = {
            "ref": f"T-{uuid.uuid4().hex[:10]}",
            "persona": "test",
            "full_name": "Test Person",
            "dob": date(1990, 1, 1),
            "balance_jpy": 100_000,
        } | overrides
        async with db_engine.begin() as conn:
            await insert_debtor(conn, SeedDebtor(**fields))
        return str(fields["ref"])

    return _make
