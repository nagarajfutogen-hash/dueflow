from datetime import date
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.models import Conversation, Debtor, Event, PromiseStatus
from app.db.seed import SeedPromise
from tests.conftest import MakeDebtor

NOON_JST = "2026-10-24T12:00:00+09:00"
LATE_JST = "2026-10-24T22:00:00+09:00"


async def _dial(client: httpx.AsyncClient, ref: str, **overrides: Any) -> httpx.Response:
    body = {"debtor_ref": ref, "sim_time": NOON_JST, "target": "mobile", "agent_version": "v1"}
    return await client.post("/calls", json=body | overrides)


async def _events_for(engine: AsyncEngine, event_ids: list[int]) -> list[Any]:
    async with engine.connect() as conn:
        stmt = select(Event.conversation_id, Event.debtor_id, Event.rule, Event.decision).where(
            Event.id.in_(event_ids)
        )
        return list((await conn.execute(stmt)).all())


async def _conversation_count(engine: AsyncEngine, ref: str) -> int:
    async with engine.connect() as conn:
        rows = await conn.execute(
            select(Conversation.id).join(Debtor).where(Debtor.external_ref == ref)
        )
        return len(rows.all())


async def test_allowed_dial_creates_conversation_with_linked_events(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    ref = await make_debtor()
    resp = await _dial(api_client, ref)
    assert resp.status_code == 200
    body = resp.json()
    assert body["allowed"] is True
    assert body["conversation_id"] is not None
    assert [d["rule"] for d in body["decisions"]] == [
        "stop_contact",
        "contact_window",
        "attempt_limits",
        "workplace_contact",
    ]
    events = await _events_for(app_engine, body["event_ids"])
    assert len(events) == 4
    assert {e.conversation_id for e in events} == {body["conversation_id"]}
    assert {e.decision for e in events} == {"allowed"}


async def test_blocked_dial_creates_no_conversation_but_logs_events(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    ref = await make_debtor()
    body = (await _dial(api_client, ref, sim_time=LATE_JST)).json()
    assert body["allowed"] is False
    assert body["conversation_id"] is None
    blocked = [d for d in body["decisions"] if not d["allowed"]]
    assert [d["rule"] for d in blocked] == ["contact_window"]
    events = await _events_for(app_engine, body["event_ids"])
    assert len(events) == 4
    assert all(e.conversation_id is None and e.debtor_id is not None for e in events)
    assert await _conversation_count(app_engine, ref) == 0


async def test_second_dial_same_local_day_blocked(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    ref = await make_debtor()
    assert (await _dial(api_client, ref)).json()["allowed"] is True
    second = (await _dial(api_client, ref, sim_time="2026-10-24T15:00:00+09:00")).json()
    assert second["allowed"] is False
    assert "daily_limit_reached" in second["decisions"][2]["codes"]


async def test_blocked_dial_does_not_count_as_attempt(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    ref = await make_debtor()
    assert (await _dial(api_client, ref, sim_time=LATE_JST)).json()["allowed"] is False
    assert (await _dial(api_client, ref, sim_time="2026-10-24T09:00:00+09:00")).json()[
        "allowed"
    ] is True


async def test_stop_contact_debtor_blocked(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    ref = await make_debtor(stop_contact=True)
    body = (await _dial(api_client, ref)).json()
    assert body["allowed"] is False
    assert body["decisions"][0]["codes"] == ["stop_contact_requested"]


async def test_work_target_without_number_is_422(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    ref = await make_debtor(work=None)
    assert (await _dial(api_client, ref, target="work")).status_code == 422


async def test_work_dial_needs_justification_and_no_flag(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    unjustified = await make_debtor(work="03-0000-1111")
    refused = await make_debtor(
        work="03-0000-2222", no_workplace_contact=True, workplace_justification="x"
    )
    justified = await make_debtor(work="03-0000-3333", workplace_justification="no other number")
    assert (await _dial(api_client, unjustified, target="work")).json()["allowed"] is False
    assert (await _dial(api_client, refused, target="work")).json()["allowed"] is False
    assert (await _dial(api_client, justified, target="work")).json()["allowed"] is True


async def test_non_jst_debtor_uses_own_timezone(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    # 22:00 JST = 09:00 EDT.
    tokyo = await make_debtor()
    new_york = await make_debtor(timezone="America/New_York")
    assert (await _dial(api_client, tokyo, sim_time=LATE_JST)).json()["allowed"] is False
    assert (await _dial(api_client, new_york, sim_time=LATE_JST)).json()["allowed"] is True


async def test_snapshot_has_account_facts_but_no_pii(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    ref = await make_debtor(
        full_name="Secret Name",
        open_dispute="says already paid",
        promises=(SeedPromise(3, 20_000, 30_000, date(2026, 8, 1), PromiseStatus.BROKEN),),
        payments_jpy=(5_000, 7_000),
    )
    conversation_id = (await _dial(api_client, ref)).json()["conversation_id"]
    async with app_engine.connect() as conn:
        snapshot = await conn.scalar(
            select(Conversation.account_snapshot).where(Conversation.id == conversation_id)
        )
    assert snapshot is not None
    assert snapshot["has_open_dispute"] is True
    assert snapshot["has_broken_promise"] is True
    assert snapshot["payments_total_jpy"] == 12_000
    assert snapshot["balance_jpy"] == 100_000
    assert "Secret Name" not in str(snapshot)
    assert "1990-01-01" not in str(snapshot)


async def test_unknown_debtor_is_404(api_client: httpx.AsyncClient) -> None:
    assert (await _dial(api_client, "NOPE")).status_code == 404


async def test_naive_sim_time_is_422(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    ref = await make_debtor()
    assert (await _dial(api_client, ref, sim_time="2026-10-24T12:00:00")).status_code == 422


async def test_missing_or_wrong_token_is_401(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    ref = await make_debtor()
    body = {"debtor_ref": ref, "sim_time": NOON_JST, "target": "mobile", "agent_version": "v1"}
    no_auth = await api_client.post("/calls", json=body, headers={"Authorization": ""})
    wrong = await api_client.post("/calls", json=body, headers={"Authorization": "Bearer nope"})
    assert no_auth.status_code == 401
    assert wrong.status_code == 401


async def test_health_needs_no_token(api_client: httpx.AsyncClient) -> None:
    assert (await api_client.get("/health", headers={"Authorization": ""})).status_code == 200
