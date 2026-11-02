from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, insert, select
from sqlalchemy.ext.asyncio import AsyncConnection

from app.db.models import Speaker, Turn


@dataclass(frozen=True)
class LlmCallInfo:
    model: str
    prompt_version: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


async def append_turn(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    speaker: Speaker,
    text: str,
    started_at: datetime,
    ended_at: datetime | None = None,
    llm: LlmCallInfo | None = None,
    latency_ms: dict[str, Any] | None = None,
) -> int:
    """Append the next turn (seq = previous + 1) and return its seq.

    One session writes to a conversation at a time; the (conversation_id, seq) unique
    constraint turns any concurrent write into an error instead of a silent reorder.
    """
    last = await conn.scalar(
        select(func.max(Turn.seq)).where(Turn.conversation_id == conversation_id)
    )
    seq = (last or 0) + 1
    await conn.execute(
        insert(Turn).values(
            conversation_id=conversation_id,
            seq=seq,
            speaker=speaker,
            text=text,
            started_at=started_at,
            ended_at=ended_at,
            llm_model=llm.model if llm else None,
            prompt_version=llm.prompt_version if llm else None,
            prompt_tokens=llm.prompt_tokens if llm else None,
            completion_tokens=llm.completion_tokens if llm else None,
            latency_ms=latency_ms,
        )
    )
    return seq
