from collections.abc import AsyncIterator
from dataclasses import replace

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import load_settings
from app.db.session import make_engine
from app.main import app


async def _client_with(engine: AsyncEngine) -> AsyncIterator[httpx.AsyncClient]:
    app.state.engine = engine
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await engine.dispose()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async for c in _client_with(make_engine(load_settings().app_url)):
        yield c


@pytest.fixture
async def client_db_down() -> AsyncIterator[httpx.AsyncClient]:
    # Port 1 on localhost refuses connections immediately.
    settings = replace(load_settings(), postgres_host="127.0.0.1", postgres_port=1)
    async for c in _client_with(make_engine(settings.app_url)):
        yield c


async def test_health_ok(client: httpx.AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "db": "ok"}


async def test_health_db_unreachable(client_db_down: httpx.AsyncClient) -> None:
    resp = await client_db_down.get("/health")
    assert resp.status_code == 503
    assert resp.json() == {"status": "degraded", "db": "unreachable"}
