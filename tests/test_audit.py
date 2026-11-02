from sqlalchemy import RowMapping, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.db.audit import record_decision, record_decisions
from app.db.models import Event
from app.policy.config import load_rules
from app.policy.engine import DialTarget, check_contact_window, check_plan_terms, pre_call_check
from tests.policy.conftest import jst

RULES = load_rules()
NOW = jst(2026, 10, 24, 12)


async def _event(conn: AsyncConnection, event_id: int) -> RowMapping:
    stmt = select(
        Event.type,
        Event.rule,
        Event.decision,
        Event.reason,
        Event.input,
        Event.agent_version,
        Event.clock_time,
    ).where(Event.id == event_id)
    return (await conn.execute(stmt)).one()._mapping


async def test_allowed_decision_is_recorded(app_engine: AsyncEngine) -> None:
    decision = check_contact_window(NOW, "Asia/Tokyo", RULES)
    async with app_engine.connect() as conn:
        row = await _event(conn, await record_decision(conn, decision, agent_version="v1"))
        await conn.rollback()
    assert row["type"] == "policy_decision"
    assert row["rule"] == "contact_window"
    assert row["decision"] == "allowed"
    assert row["reason"] == decision.reason
    assert row["agent_version"] == "v1"
    assert row["clock_time"] == NOW
    assert row["input"]["codes"] == []
    assert row["input"]["rules_version"] == RULES.version


async def test_blocked_decision_is_recorded_with_codes(app_engine: AsyncEngine) -> None:
    decision = check_plan_terms(
        NOW,
        balance_jpy=100_000,
        months=7,
        down_payment_jpy=0,
        has_broken_promise=False,
        rules=RULES,
    )
    async with app_engine.connect() as conn:
        row = await _event(conn, await record_decision(conn, decision, agent_version="v1"))
        await conn.rollback()
    assert row["decision"] == "blocked"
    assert row["input"]["codes"] == ["months_out_of_range", "down_payment_too_low"]


async def test_every_pre_call_decision_is_recorded(app_engine: AsyncEngine) -> None:
    decisions = pre_call_check(
        jst(2026, 10, 24, 22),
        debtor_tz="Asia/Tokyo",
        prior_attempts=[],
        stop_contact=False,
        target=DialTarget.MOBILE,
        no_workplace_contact=False,
        workplace_justification=None,
        rules=RULES,
    )
    async with app_engine.connect() as conn:
        ids = await record_decisions(conn, decisions, agent_version="v1")
        rows = [await _event(conn, i) for i in ids]
        await conn.rollback()
    assert [r["rule"] for r in rows] == [d.rule.value for d in decisions]
    assert [r["decision"] for r in rows] == ["allowed", "blocked", "allowed", "allowed"]
