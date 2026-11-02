"""Admin dashboard (PLANNING §16): login, plain-word summaries, counts, live calls, no DOBs."""

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from app.admin import (
    BUSINESS_TZ,
    HIDDEN_IDENTITY_TEXT,
    LIVE_WINDOW,
    CallFacts,
    _per_day,
    _period_start,
    _previous_window,
    summarize,
)
from app.agent import scripts
from app.agent.actions import request_handoff
from app.agent.identity import verify_identity
from app.agent.negotiation import propose_terms, record_promise
from app.clock import FixedClock, SystemClock
from app.db.models import HandoffReason, Outcome, Speaker
from app.db.turns import append_turn
from app.main import app
from app.policy.config import load_rules
from tests.conftest import MakeDebtor

RULES = load_rules()
NOW = datetime(2026, 10, 24, 3, tzinfo=UTC)  # 12:00 JST, the simulated call time
NAME = "Mika Admin"
DOB = date(1984, 6, 1)
BALANCE = 100_000
V = "v1"
PASSWORD = "admin-test-password"


@pytest.fixture
async def admin_client(api_client: httpx.AsyncClient) -> AsyncIterator[httpx.AsyncClient]:
    """API client with an admin password set and a clean login counter."""
    saved = app.state.settings
    app.state.settings = replace(saved, admin_password=PASSWORD)
    app.state.login_tries = {}
    app.state.admin_clock = None
    yield api_client
    app.state.settings = saved
    app.state.admin_clock = None


async def _call(client: httpx.AsyncClient, make_debtor: MakeDebtor, **body: Any) -> int:
    ref = await make_debtor(full_name=NAME, dob=DOB, balance_jpy=BALANCE)
    payload = {
        "debtor_ref": ref,
        "sim_time": "2026-10-24T12:00:00+09:00",
        "target": "mobile",
        "agent_version": V,
        "label": "HUMAN_TEST",
    } | body
    return int((await client.post("/calls", json=payload)).json()["conversation_id"])


async def _say(engine: AsyncEngine, cid: int, speaker: Speaker, text: str) -> None:
    async with engine.begin() as conn:
        await append_turn(conn, conversation_id=cid, speaker=speaker, text=text, started_at=NOW)


async def _verify(engine: AsyncEngine, cid: int) -> None:
    said = f"It's {NAME}, born 1984-06-01."
    await _say(engine, cid, Speaker.DEBTOR, said)
    async with engine.begin() as conn:
        await verify_identity(
            conn, conversation_id=cid, debtor_text=said, now=NOW, rules=RULES, agent_version=V
        )
    await _say(engine, cid, Speaker.AGENT, scripts.VERIFIED_WARM)


async def _promise(engine: AsyncEngine, cid: int) -> None:
    async with engine.begin() as conn:
        result = await propose_terms(
            conn,
            conversation_id=cid,
            months=3,
            down_payment_jpy=10_000,
            now=NOW,
            rules=RULES,
            agent_version=V,
        )
        assert result.terms_id is not None
        await record_promise(
            conn, conversation_id=cid, terms_id=result.terms_id, now=NOW, agent_version=V
        )


async def _handoff(engine: AsyncEngine, cid: int) -> None:
    async with engine.begin() as conn:
        await request_handoff(
            conn,
            conversation_id=cid,
            reason=HandoffReason.HARDSHIP,
            summary="Lost their job; cannot manage any plan.",
            now=NOW,
            agent_version=V,
        )


async def _get(client: httpx.AsyncClient, path: str) -> Any:
    res = await client.get(path)
    assert res.status_code == 200, res.text
    return res.json()


# --- Login ------------------------------------------------------------------------------------


async def test_login_with_the_right_password_returns_the_api_token(
    admin_client: httpx.AsyncClient,
) -> None:
    res = await admin_client.post("/auth/login", json={"password": PASSWORD})
    assert res.status_code == 200
    assert res.json() == {"token": app.state.settings.api_token}


async def test_login_with_a_wrong_password_is_refused(admin_client: httpx.AsyncClient) -> None:
    res = await admin_client.post("/auth/login", json={"password": "nope"})
    assert res.status_code == 401
    assert "token" not in res.json()


async def test_login_is_limited_to_five_tries_a_minute(admin_client: httpx.AsyncClient) -> None:
    codes = [
        (await admin_client.post("/auth/login", json={"password": "nope"})).status_code
        for _ in range(6)
    ]
    assert codes == [401] * 5 + [429]
    # The right password is refused too until the minute has passed.
    res = await admin_client.post("/auth/login", json={"password": PASSWORD})
    assert res.status_code == 429


async def test_login_is_off_without_an_admin_password(admin_client: httpx.AsyncClient) -> None:
    app.state.settings = replace(app.state.settings, admin_password="")
    res = await admin_client.post("/auth/login", json={"password": "anything"})
    assert res.status_code == 503


@pytest.mark.parametrize("path", ["/admin/overview", "/admin/calls", "/admin/calls/1"])
async def test_admin_routes_need_the_token(admin_client: httpx.AsyncClient, path: str) -> None:
    res = await admin_client.get(path, headers={"Authorization": ""})
    assert res.status_code == 401


# --- Call detail -----------------------------------------------------------------------------


async def test_call_detail_summary_and_moments_in_plain_words(
    admin_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(admin_client, make_debtor)
    await _verify(app_engine, cid)
    await _promise(app_engine, cid)

    detail = await _get(admin_client, f"/admin/calls/{cid}")
    call = detail["call"]
    assert call["result"] == Outcome.PROMISE.value
    assert call["result_words"] == "Agreed to pay"
    assert call["label"] == "HUMAN_TEST"
    assert call["summary"] == (
        "Identity confirmed. Agreed to ¥10,000 down by October 31st, 2026, "
        "then 3 monthly payments of ¥30,000."
    )
    texts = [m["text"] for m in detail["moments"]]
    assert texts[0] == "Identity confirmed"
    assert texts[1].startswith("Offered: ¥10,000 down")
    assert texts[-1] == "Call ended: Agreed to pay"
    assert detail["balance_jpy"] == BALANCE


async def test_dob_never_appears_in_transcript_or_mentioned_dates(
    admin_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(admin_client, make_debtor)
    await _verify(app_engine, cid)
    await _say(app_engine, cid, Speaker.DEBTOR, "My payday is November 25th, 2026.")
    await _say(app_engine, cid, Speaker.DEBTOR, "Again, I was born 1984-06-01.")

    detail = await _get(admin_client, f"/admin/calls/{cid}")
    customer = [t["text"] for t in detail["transcript"] if t["who"] == "Customer"]
    assert customer == [
        HIDDEN_IDENTITY_TEXT,
        "My payday is November 25th, 2026.",
        HIDDEN_IDENTITY_TEXT,
    ]
    assert [d["date"] for d in detail["dates_mentioned"]] == ["2026-11-25"]
    assert "1984" not in str(detail)


async def test_dates_before_verification_are_hidden_and_not_listed(
    admin_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(admin_client, make_debtor)
    await _say(app_engine, cid, Speaker.DEBTOR, "Who is this? I'm born 1990-02-03.")

    detail = await _get(admin_client, f"/admin/calls/{cid}")
    assert [t["text"] for t in detail["transcript"] if t["who"] == "Customer"] == [
        HIDDEN_IDENTITY_TEXT
    ]
    assert detail["dates_mentioned"] == []


async def test_unknown_call_is_404(admin_client: httpx.AsyncClient) -> None:
    assert (await admin_client.get("/admin/calls/999999999")).status_code == 404


# --- Overview ------------------------------------------------------------------------------


async def test_overview_counts_follow_the_stored_calls(
    admin_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    # Due-soon window: the down payment (October 31st) is within 7 days of October 28th.
    app.state.admin_clock = FixedClock(datetime(2026, 10, 28, 3, tzinfo=UTC))
    before = await _get(admin_client, "/admin/overview?period=all&source=manual")

    promised = await _call(admin_client, make_debtor)
    await _verify(app_engine, promised)
    await _promise(app_engine, promised)
    handed = await _call(admin_client, make_debtor)
    await _verify(app_engine, handed)
    await _handoff(app_engine, handed)
    await _call(admin_client, make_debtor)  # never verified, never ended

    after = await _get(admin_client, "/admin/overview?period=all&source=manual")

    def delta(key: str) -> int:
        return int(after[key]) - int(before[key])

    def step(o: dict[str, Any], name: str) -> int:
        return next(int(s["count"]) for s in o["funnel"] if s["step"] == name)

    assert delta("calls") == 3
    assert delta("agreed_to_pay") == 1
    assert delta("sent_to_team") == 1
    assert delta("promised_jpy") == BALANCE
    assert step(after, "Right person confirmed") - step(before, "Right person confirmed") == 2
    assert step(after, "Payment option offered") - step(before, "Payment option offered") == 1
    assert after["labels"]["HUMAN_TEST"] - before["labels"].get("HUMAN_TEST", 0) == 3
    assert after["due_soon"]["count"] - before["due_soon"]["count"] == 1
    assert after["due_soon"]["amount_jpy"] - before["due_soon"]["amount_jpy"] == 10_000


async def test_eval_runs_are_left_out_by_default(
    admin_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    cid = await _call(admin_client, make_debtor, scenario_id="base/admin-test", label="SYNTHETIC")
    manual = await _get(admin_client, "/admin/calls")
    evals = await _get(admin_client, "/admin/calls?source=eval")
    assert cid not in [c["id"] for c in manual]
    row = next(c for c in evals if c["id"] == cid)
    assert row["source"] == "eval"
    assert row["label"] == "SYNTHETIC"


# --- Live calls ------------------------------------------------------------------------------


async def test_a_call_without_an_ending_is_live_then_not_finished(
    admin_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    cid = await _call(admin_client, make_debtor)
    live = await _get(admin_client, "/admin/calls?status=live")
    row = next(c for c in live if c["id"] == cid)
    assert row["status"] == "live"
    assert row["result_words"] == "In progress"
    assert (await _get(admin_client, "/admin/overview?period=all"))["in_progress"] >= 1

    later = SystemClock().now() + LIVE_WINDOW + timedelta(minutes=1)
    app.state.admin_clock = FixedClock(later)
    assert cid not in [c["id"] for c in await _get(admin_client, "/admin/calls?status=live")]
    row = next(c for c in await _get(admin_client, "/admin/calls") if c["id"] == cid)
    assert row["status"] == "unfinished"
    assert row["result_words"] == "Not finished"


# --- Summaries (pure) ------------------------------------------------------------------------


def test_summary_for_an_unverified_ended_call_uses_the_result_words() -> None:
    assert summarize(Outcome.END_WRONG_PERSON, "ended", CallFacts()) == "Not the right person."


def test_summary_for_stop_contact_is_said_once() -> None:
    f = CallFacts(verified=True)
    assert summarize(Outcome.STOP_CONTACT, "ended", f) == (
        "Identity confirmed. Asked not to be called again."
    )


def test_summary_for_a_payment_claim_with_no_match() -> None:
    f = CallFacts(verified=True, payment_claims=[False], handoff_reason="dispute")
    assert summarize(Outcome.HANDOFF, "ended", f) == (
        "Identity confirmed. Said they already paid: no matching payment found. "
        "Sent to team: disputes the debt or says it is paid."
    )


def test_summary_for_a_live_call() -> None:
    assert summarize(None, "live", CallFacts()) == "Call in progress."


# --- Trend: calls per day and the previous period -----------------------------------------


async def test_per_day_and_previous_period_follow_new_calls(
    admin_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    app.state.admin_clock = SystemClock()
    before = await _get(admin_client, "/admin/overview?period=today&source=manual")

    promised = await _call(admin_client, make_debtor)
    await _verify(app_engine, promised)
    await _promise(app_engine, promised)
    handed = await _call(admin_client, make_debtor)
    await _verify(app_engine, handed)
    await _handoff(app_engine, handed)
    await _call(admin_client, make_debtor)

    after = await _get(admin_client, "/admin/overview?period=today&source=manual")
    today = after["per_day"][-1]
    old = (
        before["per_day"][-1] if before["per_day"] else {"agreed": 0, "sent_to_team": 0, "other": 0}
    )
    assert len(after["per_day"]) == 1  # 'today' is one business day
    assert today["day"] == SystemClock().now().astimezone(BUSINESS_TZ).date().isoformat()
    assert (today["agreed"], today["sent_to_team"], today["other"]) == (
        old["agreed"] + 1,
        old["sent_to_team"] + 1,
        old["other"] + 1,
    )
    # Yesterday's numbers don't move when today's calls are added.
    assert after["previous"] == before["previous"]
    assert set(after["previous"]) == {"calls", "agreed_to_pay", "sent_to_team", "promised_jpy"}


async def test_all_time_has_no_previous_period(admin_client: httpx.AsyncClient) -> None:
    assert (await _get(admin_client, "/admin/overview?period=all"))["previous"] is None


def test_previous_window_has_the_same_length() -> None:
    now = datetime(2026, 10, 31, 3, tzinfo=UTC)  # 12:00 JST
    start = _period_start("7d", now)
    assert start is not None
    assert _previous_window("7d", start, now) == (start - timedelta(days=7), start)
    today = _period_start("today", now)
    assert today is not None
    assert _previous_window("today", today, now) == (today - timedelta(days=1), today)


def test_per_day_fills_quiet_days_with_zeros() -> None:
    now = datetime(2026, 10, 31, 3, tzinfo=UTC)
    rows = [SimpleNamespace(id=1, created_at=datetime(2026, 10, 29, 1, tzinfo=UTC))]
    facts = {1: CallFacts(handoff_reason="hardship")}
    days = _per_day(rows, facts, None, now)
    assert days == [
        {"day": "2026-10-29", "agreed": 0, "sent_to_team": 1, "other": 0},
        {"day": "2026-10-30", "agreed": 0, "sent_to_team": 0, "other": 0},
        {"day": "2026-10-31", "agreed": 0, "sent_to_team": 0, "other": 0},
    ]
