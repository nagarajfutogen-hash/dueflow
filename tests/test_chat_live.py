"""Live smoke tests (Groq). Not evals: one scripted run per scenario.

Runs against the server already running in the api container (dev DB), with fresh
SYNTHETIC debtors. Run:  docker compose exec -e RUN_LIVE=1 api pytest -s tests/test_chat_live.py
Writes /tmp/live_report.json (inside the container) with outcome, events and token usage.
"""

import asyncio
import json
import os
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
import websockets
from sqlalchemy import select

from app.config import load_settings
from app.db.models import Conversation, Dispute, Event, Turn
from app.db.seed import SeedDebtor, insert_debtor
from app.db.session import make_engine

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE") != "1", reason="live Groq calls; set RUN_LIVE=1"
)

BASE = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"
DOB = date(1993, 8, 8)
FIRST_REPLY_TIMEOUT_S = 40.0
IDLE_TIMEOUT_S = 8.0
REPORT = Path("/tmp/live_report.json")

SCENARIOS: dict[str, tuple[str, list[str]]] = {
    "promise": (
        "promise",
        [
            "Yes, this is {name}.",
            "My full name is {name} and I was born on 1993-08-08.",
            "Could I pay 10,000 yen now and the rest over 3 months?",
            "Yes, that works for me.",
        ],
    ),
    "wrong_person": (
        "end_wrong_person",
        [
            "Yes, speaking.",
            "{name}, born 1990-01-01.",
            "Sorry, it's {name}, born 1991-02-02.",
        ],
    ),
    "third_party": ("third_party_callback", ["They're not home right now. I'm their roommate."]),
    "stop_contact": ("stop_contact", ["Please stop calling this number. Do not contact me again."]),
    "dispute_handoff": (
        "handoff",
        [
            "Yes, that's me.",
            "{name}, date of birth 1993-08-08.",
            "I don't owe anything. I already paid this loan back last year.",
        ],
    ),
}


async def _new_debtor(name: str) -> str:
    ref = f"LIVE-{uuid.uuid4().hex[:8]}"
    engine = make_engine(load_settings().owner_url)
    async with engine.begin() as conn:
        await insert_debtor(
            conn,
            SeedDebtor(ref=ref, persona="live_smoke", full_name=name, dob=DOB, balance_jpy=100_000),
        )
    await engine.dispose()
    return ref


async def _drain(ws: Any, first_timeout: float) -> list[dict[str, Any]]:
    """Collect server messages until the agent goes quiet or the call ends."""
    messages: list[dict[str, Any]] = []
    timeout = first_timeout
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout)
        except (TimeoutError, websockets.ConnectionClosed):
            return messages
        msg = json.loads(raw)
        messages.append(msg)
        if msg["type"] == "end":
            return messages
        timeout = IDLE_TIMEOUT_S


async def _trail(conversation_id: int) -> dict[str, Any]:
    engine = make_engine(load_settings().app_url)
    async with engine.connect() as conn:
        outcome = await conn.scalar(
            select(Conversation.outcome).where(Conversation.id == conversation_id)
        )
        events = [
            {
                "type": e.type,
                "rule": e.rule,
                "decision": e.decision,
                "reason": e.reason,
                "detail": e.input.get("detail") or e.input.get("to"),
            }
            for e in await conn.execute(
                select(Event.type, Event.rule, Event.decision, Event.reason, Event.input)
                .where(Event.conversation_id == conversation_id)
                .order_by(Event.id)
            )
        ]
        turns = [
            f"{t.speaker}: {t.text}"
            for t in await conn.execute(
                select(Turn.speaker, Turn.text)
                .where(Turn.conversation_id == conversation_id)
                .order_by(Turn.seq)
            )
        ]
        disputes = (
            await conn.execute(
                select(Dispute.status)
                .join(Conversation, Conversation.debtor_id == Dispute.debtor_id)
                .where(Conversation.id == conversation_id)
            )
        ).scalars()
        dispute_statuses = [str(d) for d in disputes]
    await engine.dispose()
    return {
        "outcome": str(outcome) if outcome else None,
        "events": events,
        "turns": turns,
        "disputes": dispute_statuses,
    }


async def _run(scenario: str) -> dict[str, Any]:
    expected, lines = SCENARIOS[scenario]
    name = f"Live {scenario.title().replace('_', '')}"
    token = os.environ["API_TOKEN"]
    ref = await _new_debtor(name)
    async with httpx.AsyncClient(base_url=BASE, timeout=30) as client:
        dial = await client.post(
            "/calls",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "debtor_ref": ref,
                "sim_time": "2026-10-24T12:00:00+09:00",
                "target": "mobile",
                "agent_version": "v1",
                "scenario_id": f"smoke:{scenario}",
            },
        )
    conversation_id = int(dial.json()["conversation_id"])
    transcript: list[dict[str, Any]] = []
    async with websockets.connect(f"{WS_BASE}/chat/ws?conversation_id={conversation_id}") as ws:
        await ws.send(json.dumps({"type": "auth", "token": token}))
        transcript += await _drain(ws, FIRST_REPLY_TIMEOUT_S)
        for line in lines:
            if any(m["type"] == "end" for m in transcript):
                break
            await ws.send(json.dumps({"type": "user", "text": line.format(name=name)}))
            transcript.append({"type": "debtor", "text": line.format(name=name)})
            transcript += await _drain(ws, FIRST_REPLY_TIMEOUT_S)
    usage = [m for m in transcript if m["type"] == "usage"]
    trail = await _trail(conversation_id)
    return {
        "scenario": scenario,
        "conversation_id": conversation_id,
        "expected": expected,
        "passed": trail["outcome"] == expected,
        "usage": usage[-1] if usage else None,
        "transcript": [m for m in transcript if m["type"] != "usage"],
        **trail,
    }


def _save(result: dict[str, Any]) -> None:
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    report[result["scenario"]] = result
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_live_smoke(scenario: str) -> None:
    result = asyncio.run(_run(scenario))
    _save(result)
    print(f"\n[{scenario}] outcome={result['outcome']} expected={result['expected']}")
    assert result["passed"], json.dumps(result["events"], indent=1)
    if scenario == "dispute_handoff":
        assert "open" in result["disputes"]
