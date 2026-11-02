"""Agent v5 callbacks (V5-3): saved from the customer's own words, dialed only once the owner
approves, and the normal pre-call check still decides every dial.

Calls run at Saturday 24 October 2026, noon JST (tests/test_chat_offline.py); "Friday" is
30 October.
"""

from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent.history import get_account_history
from app.agent.identity import verify_identity
from app.agent.nodes import negotiate_node
from app.agent.session import ChatSession
from app.agent.tools import Handler
from app.agent.versions import load_version
from app.calls import auto_dial, auto_picks
from app.clock import FixedClock
from app.db.models import Callback, CallbackStatus, ContactPrefs, Debtor, Event, Outcome
from app.db.models import ResultLabel as Label
from app.main import app
from app.policy.config import load_rules
from tests.conftest import MakeDebtor
from tests.test_chat_offline import FM, NAME, _session, _verify

RULES = load_rules()
FRIDAY = date(2026, 10, 30)


@pytest.fixture
async def client(api_client: httpx.AsyncClient) -> AsyncIterator[httpx.AsyncClient]:
    app.state.admin_clock = FixedClock(datetime(2026, 10, 26, 3, tzinfo=UTC))  # Monday
    yield api_client
    app.state.admin_clock = None


async def _asked_for_callback(
    client: httpx.AsyncClient, make_debtor: MakeDebtor, engine: AsyncEngine, **debtor: Any
) -> tuple[ChatSession, dict[str, Handler], dict[str, Any]]:
    session, _, h = await _session(client, make_debtor, engine, "v5", **debtor)
    await _verify(session, h)
    await session.record_debtor("Honestly I get paid on Friday, can you call me then?")
    out, _ = await h["record_callback"]({"day_phrase": "Friday", "reason": "payday"}, FM)
    return session, h, out


async def _callback(engine: AsyncEngine, conversation_id: int) -> Any:
    async with engine.connect() as conn:
        return (
            await conn.execute(select(Callback).where(Callback.conversation_id == conversation_id))
        ).one_or_none()


async def _ref(engine: AsyncEngine, session: ChatSession) -> str:
    async with engine.connect() as conn:
        cb = await _callback(engine, session.conversation_id)
        ref = await conn.scalar(select(Debtor.external_ref).where(Debtor.id == cb.debtor_id))
    assert ref is not None
    return ref


async def _pick(engine: AsyncEngine, ref: str, today: date) -> Any:
    async with engine.connect() as conn:
        picks = await auto_picks(conn, RULES, today)
    seen_plain = False
    for p in picks:  # owner-approved, due callbacks always come first
        seen_plain = seen_plain or p.callback_id is None
        assert not (seen_plain and p.callback_id is not None)
    return next((p for p in picks if p.debtor_ref == ref), None)


# --- The agent saves a callback from the customer's own words ---------------------------------


async def test_v5_callback_on_the_customers_day_is_saved_as_requested(
    client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, out = await _asked_for_callback(client, make_debtor, app_engine)
    assert out == {"recorded": True, "callback_on": "October 30th"}  # V5-6: no year this year
    assert session.outcome == Outcome.CALLBACK_SCHEDULED.value
    cb = await _callback(app_engine, session.conversation_id)
    assert (cb.due_date, cb.status, cb.reason, cb.customer_words) == (
        FRIDAY,
        CallbackStatus.REQUESTED,
        "payday",
        "friday",
    )


async def test_v5_a_day_the_customer_never_said_is_refused_and_the_call_goes_on(
    client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, h = await _session(client, make_debtor, app_engine, "v5")
    await _verify(session, h)
    await session.record_debtor("I need a bit more time.")
    for phrase in ("Monday", "2026-11-02"):  # never said; a date the LLM worked out itself
        out, _ = await h["record_callback"]({"day_phrase": phrase, "reason": "needs_time"}, FM)
        assert out["recorded"] is False and "date_not_said" in out["codes"]
    assert session.outcome is None
    assert await _callback(app_engine, session.conversation_id) is None


def test_only_v5_has_the_callback_tool() -> None:
    v5 = negotiate_node({"balance_jpy": 1}, RULES, load_version("v5"))
    v4 = negotiate_node({"balance_jpy": 1}, RULES, load_version("v4"))
    assert "record_callback" in {t.name for t in v5.tools}
    assert "record_callback" not in {t.name for t in v4.tools}
    assert "exact words for the day" in v5.task


# --- Dialed only once the owner approves --------------------------------------------------------


async def test_a_requested_callback_is_never_auto_dialed_until_approved(
    client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, _ = await _asked_for_callback(client, make_debtor, app_engine)
    ref = await _ref(app_engine, session)
    # Waiting for approval: left out of auto dial completely, even on the day itself.
    assert await _pick(app_engine, ref, FRIDAY) is None

    waiting = await client.get("/admin/callbacks?status=requested")
    row = next(r for r in waiting.json() if r["customer_ref"] == ref)
    assert (row["due_words"], row["reason_words"]) == ("October 30th, 2026", "waiting for payday")

    approved = await client.post(f"/admin/callbacks/{row['id']}/approve")
    assert approved.json() == {"id": row["id"], "status": "approved"}
    assert await _pick(app_engine, ref, date(2026, 10, 29)) is None  # approved, not yet due
    pick = await _pick(app_engine, ref, FRIDAY)
    assert pick is not None and pick.callback_id == row["id"]
    assert pick.sim_time.date() == FRIDAY

    async with app_engine.begin() as conn:
        picked = await auto_dial(
            conn, RULES, today=FRIDAY, agent_version="v5", label=Label.HUMAN_TEST, debtor_ref=ref
        )
    assert picked is not None and picked[1].allowed
    cb = await _callback(app_engine, session.conversation_id)
    assert (cb.status, cb.done_conversation_id) == (
        CallbackStatus.DONE,
        picked[1].conversation_id,
    )
    assert await _pick(app_engine, ref, FRIDAY) is not None  # back in the normal queue

    # The callback call knows it is the callback they asked for.
    new_id = picked[1].conversation_id
    assert new_id is not None
    async with app_engine.begin() as conn:
        await verify_identity(
            conn,
            conversation_id=new_id,
            debtor_text=f"{NAME}, born 1993-08-08",
            now=datetime(2026, 10, 30, 3, tzinfo=UTC),
            rules=RULES,
            agent_version="v5",
        )
        history = await get_account_history(
            conn,
            conversation_id=new_id,
            now=datetime(2026, 10, 30, 3, tzinfo=UTC),
            agent_version="v5",
        )
    assert history.facts is not None
    assert history.facts["callback_they_asked_for"] == "2026-10-30"


async def test_owner_decisions_are_audited_and_cannot_be_repeated(
    client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, _ = await _asked_for_callback(client, make_debtor, app_engine)
    cb = await _callback(app_engine, session.conversation_id)
    assert (await client.post(f"/admin/callbacks/{cb.id}/reject")).status_code == 200
    again = await client.post(f"/admin/callbacks/{cb.id}/approve")
    assert again.status_code == 409 and "already rejected" in again.json()["detail"]
    async with app_engine.connect() as conn:
        tools = list(
            await conn.scalars(
                select(Event.input["tool"].astext).where(
                    Event.conversation_id == session.conversation_id,
                    Event.agent_version == "admin",
                )
            )
        )
    assert tools == ["admin_reject_callback"]
    # Rejected: never dialed as a callback, back in the normal queue.
    pick = await _pick(app_engine, await _ref(app_engine, session), FRIDAY)
    assert pick is not None and pick.callback_id is None


async def test_a_callback_whose_day_has_passed_cannot_be_approved(
    client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, _ = await _asked_for_callback(client, make_debtor, app_engine)
    cb = await _callback(app_engine, session.conversation_id)
    app.state.admin_clock = FixedClock(datetime(2026, 11, 2, 3, tzinfo=UTC))
    res = await client.post(f"/admin/callbacks/{cb.id}/approve")
    assert res.status_code == 409 and "passed" in res.json()["detail"]
    listed = await client.get("/admin/callbacks?status=requested")
    assert next(r for r in listed.json() if r["id"] == cb.id)["overdue"] is True


async def test_an_approved_callback_never_overrides_stop_contact(
    client: httpx.AsyncClient,
    make_debtor: MakeDebtor,
    app_engine: AsyncEngine,
    db_engine: AsyncEngine,
) -> None:
    session, _, _ = await _asked_for_callback(client, make_debtor, app_engine)
    cb = await _callback(app_engine, session.conversation_id)
    await client.post(f"/admin/callbacks/{cb.id}/approve")
    async with db_engine.begin() as conn:
        await conn.execute(
            update(ContactPrefs)
            .where(ContactPrefs.debtor_id == cb.debtor_id)
            .values(stop_contact=True)
        )
    assert await _pick(app_engine, await _ref(app_engine, session), FRIDAY) is None


@pytest.mark.parametrize(
    ("method", "path"),
    [("get", "/admin/callbacks"), ("post", "/admin/callbacks/1/approve")],
)
async def test_callback_routes_need_the_token(
    client: httpx.AsyncClient, method: str, path: str
) -> None:
    res = await client.request(method, path, headers={"Authorization": ""})
    assert res.status_code == 401


async def test_unknown_callback_is_404(client: httpx.AsyncClient) -> None:
    assert (await client.post("/admin/callbacks/999999999/reject")).status_code == 404


async def test_v5_refusal_wording_never_names_a_day_the_guard_would_block(
    client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    """V5-5 (calls 192, 193): no 'tomorrow' in the instruction, and the reason fits the code."""
    session, _, h = await _session(client, make_debtor, app_engine, "v5")
    await _verify(session, h)
    await session.record_debtor("Call me back on January 20th.")
    far, _ = await h["record_callback"]({"day_phrase": "January 20th", "reason": "other"}, FM)
    assert far["recorded"] is False and "too_far" in far["codes"]
    assert "next few weeks" in far["instruction"]
    await session.record_debtor("Maybe later.")
    vague, _ = await h["record_callback"]({"day_phrase": "later", "reason": "other"}, FM)
    for out in (far, vague):
        assert "tomorrow" not in out["instruction"].lower()
    assert session.outcome is None
