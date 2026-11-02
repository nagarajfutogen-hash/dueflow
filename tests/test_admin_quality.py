"""Admin dashboard D-3: the Quality view counts the guard, every rule, the ladder, callbacks."""

from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent.scripts import SAFE_FALLBACK_POST_V4
from tests.conftest import MakeDebtor
from tests.test_chat_offline import FM, _session, _verify


async def _q(client: httpx.AsyncClient, **params: str) -> Any:
    res = await client.get("/admin/quality", params={"source": "manual", **params})
    assert res.status_code == 200, res.text
    return res.json()


def _guard(q: Any, version: str) -> dict[str, int]:
    return next((g for g in q["guard"] if g["version"] == version), {}) or {
        k: 0 for k in ("calls", "checks", "blocked", "then_passed", "then_blocked", "fallbacks")
    }


def _count(rows: list[dict[str, Any]], key: str, value: str) -> int:
    return next((r["count"] for r in rows if r[key] == value), 0)


async def test_guard_counts_blocks_what_followed_and_fallbacks(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    before = _guard(await _q(api_client, version="v4"), "v4")
    session, _, h = await _session(api_client, make_debtor, app_engine, "v4")
    await _verify(session, h)
    blocked = await session.guard_and_send("You could pay ¥1,234 a month.", None, None)
    assert SAFE_FALLBACK_POST_V4 in blocked  # offline: no rewrite, the fallback is said
    await session.guard_and_send("Shall we look at a plan together?", None, None)
    after = _guard(await _q(api_client, version="v4"), "v4")
    delta = {k: after[k] - before[k] for k in after if k != "version"}
    assert delta == {
        "calls": 1,
        "checks": 2,
        "blocked": 1,
        "then_passed": 1,
        "then_blocked": 0,
        "fallbacks": 1,
    }
    reasons = (await _q(api_client, version="v4"))["block_reasons"]
    assert _count(reasons, "reason", "unapproved_figure") >= 1


async def test_ladder_offers_handoffs_and_callbacks_are_counted(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    before = await _q(api_client, version="v5")
    session, _, h = await _session(api_client, make_debtor, app_engine, "v5")
    await _verify(session, h)
    await h["next_offer"]({"reason": "first_offer"}, FM)
    await h["next_offer"]({"reason": "monthly_too_high", "affordable_monthly_jpy": 1_000}, FM)

    other, _, h2 = await _session(api_client, make_debtor, app_engine, "v5")
    await _verify(other, h2)
    await other.record_debtor("Call me on Friday please.")
    await h2["record_callback"]({"day_phrase": "Friday", "reason": "busy_now"}, FM)

    after = await _q(api_client, version="v5")

    def grew(section: list[dict[str, Any]], key: str, value: str, old: list[dict[str, Any]]) -> int:
        return _count(section, key, value) - _count(old, key, value)

    assert grew(after["ladder"]["offers"], "rung", "pay_in_full", before["ladder"]["offers"]) == 1
    assert (
        grew(
            after["ladder"]["handoffs"],
            "code",
            "below_account_floor",
            before["ladder"]["handoffs"],
        )
        == 1
    )
    assert grew(after["callbacks"], "status", "requested", before["callbacks"]) == 1
    rules = {r["rule"]: r for r in after["rules"]}
    assert rules["negotiation_ladder"]["blocked"] >= 1
    assert rules["callback_request"]["allowed"] >= 1


async def test_a_version_with_no_calls_is_empty(api_client: httpx.AsyncClient) -> None:
    q = await _q(api_client, version="v9")
    assert q["calls"] == 0 and q["guard"] == [] and q["rules"] == []


async def test_quality_needs_the_token(api_client: httpx.AsyncClient) -> None:
    res = await api_client.get("/admin/quality", headers={"Authorization": ""})
    assert res.status_code == 401
