from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from app.calls import auto_sim_time, auto_target
from app.db.models import ContactPrefs, Debtor
from app.policy.config import load_rules
from app.policy.engine import DialTarget
from tests.conftest import MakeDebtor

RULES = load_rules()
TOKYO = ZoneInfo("Asia/Tokyo")
TODAY = date(2026, 10, 24)


def _at(day: int, hour: int = 12) -> datetime:
    return datetime(2026, 10, day, hour, tzinfo=TOKYO)


def _debtor(**phones: str | None) -> Debtor:
    return Debtor(
        phone_mobile=phones.get("mobile"),
        phone_home=phones.get("home"),
        phone_work=phones.get("work"),
    )


def test_never_called_debtor_gets_noon_today_local() -> None:
    at = auto_sim_time([], "America/New_York", RULES, TODAY)
    assert at == datetime(2026, 10, 24, 12, tzinfo=ZoneInfo("America/New_York"))


def test_called_today_moves_to_the_next_day() -> None:
    assert auto_sim_time([_at(24)], "Asia/Tokyo", RULES, TODAY) == _at(25)


def test_weekly_limit_skips_ahead_until_a_slot_frees() -> None:
    prior = [_at(24), _at(25), _at(26)]  # 3 in 7 days: next allowed is after 10-24 noon + 7d
    at = auto_sim_time(prior, "Asia/Tokyo", RULES, TODAY)
    assert at == _at(31)
    assert sum(1 for a in prior if at - timedelta(days=7) < a <= at) < 3


def test_target_prefers_mobile_then_home_and_work_only_when_justified() -> None:
    assert auto_target(_debtor(mobile="1", home="2"), None) is DialTarget.MOBILE
    assert auto_target(_debtor(home="2"), None) is DialTarget.HOME
    justified = ContactPrefs(
        no_workplace_contact=False, workplace_justification="no personal number"
    )
    assert auto_target(_debtor(work="3"), justified) is DialTarget.WORK
    assert auto_target(_debtor(work="3"), ContactPrefs(no_workplace_contact=False)) is None
    banned = ContactPrefs(no_workplace_contact=True, workplace_justification="x")
    assert auto_target(_debtor(work="3"), banned) is None


async def test_auto_dial_picks_a_callable_debtor_and_records_the_check(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    await make_debtor()  # at least one never-dialed debtor exists
    resp = await api_client.post("/calls/auto", json={"agent_version": "v4"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["allowed"] is True
    assert body["conversation_id"] is not None
    assert len(body["event_ids"]) == 4  # the normal pre-call check ran and was recorded
    card = (await api_client.get(f"/debtors/{body['debtor_ref']}")).json()
    assert card["stop_contact"] is False


async def test_auto_dial_never_picks_stop_contact(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    stopped = await make_debtor(stop_contact=True)
    for _ in range(3):
        body = (await api_client.post("/calls/auto", json={"agent_version": "v4"})).json()
        assert body["debtor_ref"] != stopped


async def _auto(client: httpx.AsyncClient, ref: str) -> httpx.Response:
    return await client.post("/calls/auto", json={"agent_version": "v4", "debtor_ref": ref})


async def test_chosen_debtor_shifts_to_the_next_free_day(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    ref = await make_debtor()
    first = (await _auto(api_client, ref)).json()
    second = (await _auto(api_client, ref)).json()
    assert first["debtor_ref"] == second["debtor_ref"] == ref
    assert first["allowed"] and second["allowed"]
    gap = datetime.fromisoformat(second["sim_time"]) - datetime.fromisoformat(first["sim_time"])
    assert gap == timedelta(days=1)


async def test_chosen_stop_contact_debtor_is_dialed_and_blocked(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    body = (await _auto(api_client, await make_debtor(stop_contact=True))).json()
    assert body["allowed"] is False
    assert body["conversation_id"] is None
    assert "stop_contact" in [d["rule"] for d in body["decisions"] if not d["allowed"]]


async def test_chosen_unknown_debtor_is_404(api_client: httpx.AsyncClient) -> None:
    assert (await _auto(api_client, "NOPE-1")).status_code == 404


async def test_auto_dial_skips_open_disputes_but_a_chosen_one_can_be_dialed(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    disputed = await make_debtor(open_dispute="Says it was repaid.")
    for _ in range(3):
        body = (await api_client.post("/calls/auto", json={"agent_version": "v4"})).json()
        assert body["debtor_ref"] != disputed
    chosen = (await _auto(api_client, disputed)).json()
    assert chosen["debtor_ref"] == disputed and chosen["allowed"] is True
