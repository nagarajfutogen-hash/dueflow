"""Admin dashboard D-2: customers, payment calendar, needs attention. No DOB, masked phones."""

import json
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from app.admin_customers import mask_phone, promise_schedule
from app.clock import FixedClock
from app.db.models import PromiseKind, PromiseStatus, Speaker
from app.db.seed import SeedPayment, SeedPromise
from app.main import app
from tests.conftest import MakeDebtor
from tests.test_admin import _handoff, _promise, _say, _verify

DOB = date(1984, 6, 1)  # the DOB tests/test_admin.py verifies with
NAME = "Mika Admin"


@pytest.fixture
async def client(api_client: httpx.AsyncClient) -> AsyncIterator[httpx.AsyncClient]:
    app.state.admin_clock = FixedClock(datetime(2026, 11, 20, 3, tzinfo=UTC))
    yield api_client
    app.state.admin_clock = None


async def _customer(make_debtor: MakeDebtor, **fields: Any) -> str:
    return await make_debtor(full_name=NAME, dob=DOB, balance_jpy=100_000, **fields)


async def _call(client: httpx.AsyncClient, ref: str, **body: Any) -> int:
    payload = {
        "debtor_ref": ref,
        "sim_time": "2026-10-24T12:00:00+09:00",
        "target": "mobile",
        "agent_version": "v1",
        "label": "HUMAN_TEST",
    } | body
    return int((await client.post("/calls", json=payload)).json()["conversation_id"])


async def _get(client: httpx.AsyncClient, path: str) -> Any:
    res = await client.get(path)
    assert res.status_code == 200, res.text
    return res.json()


def _no_pii(blob: Any) -> None:
    text = json.dumps(blob, ensure_ascii=False)
    assert "1984-06-01" not in text and "1984" not in text  # date of birth
    assert "090-0000-0000" not in text  # the seeded mobile number in full


# --- Customers --------------------------------------------------------------------------------


async def test_customer_list_shows_badges_and_leaves_out_machine_refs(
    client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    disputed = await _customer(make_debtor, open_dispute="says the amount is wrong")
    stopped = await _customer(make_debtor, stop_contact=True, no_workplace_contact=True)
    machine = await make_debtor(ref="EV-adm-only-evals")
    by_hand = await make_debtor(ref="LIVE-adm-called-by-hand")
    await _call(client, by_hand)
    rows = {c["ref"]: c for c in await _get(client, "/admin/customers")}
    assert rows[disputed]["badges"] == ["Open dispute"]
    assert rows[stopped]["badges"] == ["Don't call", "Not at work"]
    assert machine not in rows  # used only by evaluations
    assert rows[by_hand]["calls"] == 1  # a machine ref someone called by hand is shown
    _no_pii(rows)


async def test_customer_page_has_the_full_story_without_dob_or_phone_numbers(
    client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    ref = await _customer(
        make_debtor,
        min_monthly_jpy=5_000,
        max_months=5,
        payment_records=(SeedPayment(20_000, date(2026, 10, 1), "card", "CARD-1"),),
    )
    cid = await _call(client, ref)
    await _verify(app_engine, cid)
    await _say(app_engine, cid, Speaker.DEBTOR, "My payday is November 25th, 2026.")
    await _promise(app_engine, cid)

    c = await _get(client, f"/admin/customers/{ref}")
    _no_pii(c)
    assert c["phones"]["mobile"] == "•••000"
    assert c["plan_limits"] == {"min_monthly_jpy": 5_000, "max_months": 5}
    assert c["paid_total_jpy"] == 20_000 and c["payments"][0]["reference"] == "CARD-1"
    (plan,) = c["plans"]
    assert plan["status_words"] == "Active" and not plan["partial_schedule"]
    assert plan["total_jpy"] == 100_000  # every plan collects the full balance
    assert [p["what"] for p in plan["payments"]] == [
        "Down payment",
        "Monthly payment 1 of 3",
        "Monthly payment 2 of 3",
        "Monthly payment 3 of 3",
    ]
    assert [x["result_words"] for x in c["calls"]] == ["Agreed to pay"]
    assert [d["date"] for d in c["dates_mentioned"]] == ["2026-11-25"]


async def test_unknown_or_evaluation_only_customer_is_404(
    client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    machine = await make_debtor(ref="EV-adm-hidden")
    for ref in ("NOPE-1", machine):
        assert (await client.get(f"/admin/customers/{ref}")).status_code == 404


# --- Payment calendar --------------------------------------------------------------------------


def test_seeded_plans_without_a_stored_schedule_are_partial() -> None:
    items, partial = promise_schedule(
        3, 20_000, date(2026, 11, 30), PromiseKind.INSTALLMENT_PLAN, None
    )
    assert partial
    assert [(i["due"], i["what"]) for i in items] == [
        ("2026-11-30", "Monthly payment 1 of 3"),
        ("2026-12-30", "Monthly payment 2 of 3"),
        ("2027-01-30", "Monthly payment 3 of 3"),
    ]


async def test_calendar_lists_payments_and_says_when_a_due_date_passed_unpaid(
    client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    plan = SeedPromise(3, 10_000, 30_000, date(2026, 11, 10), PromiseStatus.ACTIVE)
    unpaid = await _customer(make_debtor, promises=(plan,))
    paid = await _customer(
        make_debtor,
        promises=(plan,),
        payment_records=(SeedPayment(30_000, date(2026, 11, 12)),),
    )
    days = await _get(client, "/admin/calendar?start=2026-11-01&days=30")
    nov10 = next(d for d in days if d["day"] == "2026-11-10")
    items = {i["customer_ref"]: i for i in nov10["items"]}
    assert items[unpaid]["what"] == "Monthly payment 1 of 3"
    assert items[unpaid]["note"] == "Due date passed, no payment on file since"
    assert items[unpaid]["partial_schedule"] is True
    assert items[paid]["note"] is None  # a payment is on file since the due date


def test_masked_phone_keeps_only_the_last_three_digits() -> None:
    assert mask_phone("+1-555-0100") == "•••100"
    assert mask_phone(None) is None


# --- Needs attention --------------------------------------------------------------------------


async def test_needs_attention_lists_the_team_queue_and_open_disputes(
    client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    app.state.admin_clock = None  # handoffs are stamped with the real time
    ref = await _customer(make_debtor)
    cid = await _call(client, ref)
    await _verify(app_engine, cid)
    await _handoff(app_engine, cid)
    disputed = await _customer(make_debtor, open_dispute="says the amount is wrong")
    att = await _get(client, "/admin/attention")
    team = next(h for h in att["sent_to_team"] if h["call"] == cid)
    assert team["reason_words"] == "can't afford any allowed plan"
    assert team["note"] == "Lost their job; cannot manage any plan."
    assert any(d["customer_ref"] == disputed for d in att["open_disputes"])
    _no_pii(att)


@pytest.mark.parametrize(
    "path", ["/admin/customers", "/admin/customers/D-001", "/admin/calendar", "/admin/attention"]
)
async def test_d2_routes_need_the_token(client: httpx.AsyncClient, path: str) -> None:
    assert (await client.get(path, headers={"Authorization": ""})).status_code == 401
