from datetime import UTC, date, datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent import scripts
from app.agent.identity import verify_identity
from app.agent.negotiation import propose_terms
from app.agent.output_guard import guard_agent_reply
from app.db.models import Event, PromiseStatus
from app.db.seed import SeedPromise
from app.policy.config import load_rules
from tests.conftest import MakeDebtor

RULES = load_rules()
NOW = datetime(2026, 10, 24, 3, tzinfo=UTC)
NAME = "Rina Guardtest"
V = "v1"


async def _call(
    client: httpx.AsyncClient, make_debtor: MakeDebtor, engine: AsyncEngine, **debtor: Any
) -> int:
    ref = await make_debtor(full_name=NAME, dob=date(1991, 3, 3), **debtor)
    body = {
        "debtor_ref": ref,
        "sim_time": "2026-10-24T12:00:00+09:00",
        "target": "mobile",
        "agent_version": V,
    }
    return int((await client.post("/calls", json=body)).json()["conversation_id"])


async def _verify(engine: AsyncEngine, cid: int) -> None:
    async with engine.begin() as conn:
        await verify_identity(
            conn,
            conversation_id=cid,
            debtor_text=f"{NAME}, born 1991-03-03",
            now=NOW,
            rules=RULES,
            agent_version=V,
        )


async def _guard(engine: AsyncEngine, cid: int, text: str) -> str:
    async with engine.begin() as conn:
        return await guard_agent_reply(
            conn, conversation_id=cid, text=text, now=NOW, rules=RULES, agent_version=V
        )


async def _output_events(engine: AsyncEngine, cid: int) -> list[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            select(Event.decision)
            .where(Event.conversation_id == cid, Event.rule == "output_check")
            .order_by(Event.id)
        )
        return [r.decision for r in rows]


async def test_pre_verification_disclosure_replaced_and_logged(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(api_client, make_debtor, app_engine)
    said = await _guard(app_engine, cid, "Hi! I'm calling about your overdue balance.")
    assert said == f"Hi! {scripts.SAFE_FALLBACK_PRE}"
    assert await _output_events(app_engine, cid) == ["blocked"]


async def test_one_event_per_reply_allowed_or_blocked(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(api_client, make_debtor, app_engine)
    await _verify(app_engine, cid)
    assert await _guard(app_engine, cid, "Your balance is ¥100,000.") == (
        "Your balance is ¥100,000."
    )
    blocked = await _guard(app_engine, cid, "We could accept ¥40,000 today.")
    assert blocked == scripts.SAFE_FALLBACK_POST
    assert await _output_events(app_engine, cid) == ["allowed", "blocked"]


async def test_approved_terms_text_passes_the_interceptor(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(api_client, make_debtor, app_engine, balance_jpy=123_457)
    await _verify(app_engine, cid)
    async with app_engine.begin() as conn:
        proposal = await propose_terms(
            conn,
            conversation_id=cid,
            months=4,
            down_payment_jpy=12_346,
            now=NOW,
            rules=RULES,
            agent_version=V,
        )
    assert proposal.terms is not None
    # The canonical terms text, built by code, must never trip the interceptor.
    assert await _guard(app_engine, cid, proposal.terms) == proposal.terms
    confirmation = scripts.promise_confirmed(proposal.terms)
    assert await _guard(app_engine, cid, confirmation) == confirmation


async def test_unapproved_figure_before_terms_is_blocked(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(api_client, make_debtor, app_engine)
    await _verify(app_engine, cid)
    # The debtor offered ¥5,000/month: the agent may not repeat it until approved.
    said = await _guard(app_engine, cid, "Sure, ¥5,000 a month works for us.")
    assert said == scripts.SAFE_FALLBACK_POST


async def test_history_figures_are_allowed(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(
        api_client,
        make_debtor,
        app_engine,
        promises=(SeedPromise(3, 20_000, 26_000, date(2026, 8, 1), PromiseStatus.BROKEN),),
    )
    await _verify(app_engine, cid)
    text = "Last time you agreed to ¥26,000 a month from August 1, which was missed."
    assert await _guard(app_engine, cid, text) == text


async def test_rule_limits_in_the_prompt_are_allowed(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    # Live smoke run 2026-10-24: "pay in full within 14 days" was wrongly blocked (count:14).
    cid = await _call(api_client, make_debtor, app_engine)
    await _verify(app_engine, cid)
    text = "You can pay in full within 14 days, or over 3 to 6 months with 10% down."
    assert await _guard(app_engine, cid, text) == text
