import asyncio
import logging

from sqlalchemy import URL, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 3.0


def make_engine(url: URL) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True, connect_args={"timeout": CONNECT_TIMEOUT_S})


async def ping(engine: AsyncEngine) -> bool:
    """Return True if the database answers SELECT 1 within the timeout."""
    try:
        async with asyncio.timeout(CONNECT_TIMEOUT_S), engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except (SQLAlchemyError, OSError, TimeoutError) as exc:
        # Log the error type only: the message can contain connection details.
        logger.warning("database ping failed: %s", type(exc).__name__)
        return False
    return True
