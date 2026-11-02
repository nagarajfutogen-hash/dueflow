"""Voice latency: per-turn math and the P50/P95 report over stored voice turns.

TTFB = first TTS audio byte - VAD end of speech. Perceived latency adds the VAD's stop wait
(stop_secs): the person actually stopped talking that long before VAD said so. Every number
printed comes from turns.latency_ms of voice conversations; n is always shown.

  docker compose exec api python -m app.voice.latency --site vps
"""

import argparse
import asyncio
import math
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection

from app.config import load_settings
from app.db.models import Channel, Conversation, Speaker, Turn
from app.db.session import make_engine

METRICS = ("ttfb_ms", "perceived_ms")


def turn_latency(vad_end_s: float, first_audio_s: float, vad_stop_secs: float) -> dict[str, Any]:
    """One agent turn's timings (monotonic seconds in, whole milliseconds out)."""
    ttfb = (first_audio_s - vad_end_s) * 1000
    return {
        "ttfb_ms": round(ttfb),
        "perceived_ms": round(ttfb + vad_stop_secs * 1000),
        "vad_stop_ms": round(vad_stop_secs * 1000),
    }


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (a value that was actually measured)."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


@dataclass(frozen=True)
class Summary:
    metric: str
    n: int
    p50: float
    p95: float


def summarize(rows: list[dict[str, Any]]) -> list[Summary]:
    out = []
    for metric in METRICS:
        values = [float(r[metric]) for r in rows if isinstance(r.get(metric), int | float)]
        if values:
            out.append(Summary(metric, len(values), percentile(values, 50), percentile(values, 95)))
    return out


def render(summaries: list[Summary], *, site: str, labels: set[str], interrupted: int) -> str:
    label = ", ".join(sorted(labels)) or "none"
    lines = [f"Voice latency · site: {site} · label: {label}"]
    if not summaries:
        lines.append("No measured voice turns.")
        return "\n".join(lines)
    lines.append(f"{'metric':<14}{'n':>6}{'P50 ms':>10}{'P95 ms':>10}")
    for s in summaries:
        lines.append(f"{s.metric:<14}{s.n:>6}{s.p50:>10.0f}{s.p95:>10.0f}")
    lines.append(f"Barge-ins recorded: {interrupted}")
    return "\n".join(lines)


async def voice_turn_latencies(conn: AsyncConnection) -> list[tuple[dict[str, Any], str]]:
    """latency_ms of every measured agent turn in a voice conversation, with its label."""
    rows = await conn.execute(
        select(Turn.latency_ms, Conversation.label)
        .join(Conversation, Conversation.id == Turn.conversation_id)
        .where(
            Conversation.channel == Channel.VOICE,
            Turn.speaker == Speaker.AGENT,
            Turn.latency_ms.is_not(None),
        )
    )
    # A JSON null (not SQL NULL) passes is_not(None): skip anything that is not a dict.
    return [(dict(r.latency_ms), r.label.value) for r in rows if isinstance(r.latency_ms, dict)]


async def main(site: str) -> None:
    engine = make_engine(load_settings().app_url)
    try:
        async with engine.connect() as conn:
            rows = await voice_turn_latencies(conn)
    finally:
        await engine.dispose()
    data = [r for r, _ in rows]
    interrupted = sum(1 for r in data if r.get("interrupted"))
    labels = {lbl for _, lbl in rows}
    print(render(summarize(data), site=site, labels=labels, interrupted=interrupted))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--site", required=True, choices=["vps", "local"], help="where this database measured"
    )
    asyncio.run(main(parser.parse_args().site))
