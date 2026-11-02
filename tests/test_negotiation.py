from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent import scripts
from app.agent.actions import record_no_workplace_contact, record_stop_contact, request_handoff
from app.agent.conversation import ConversationEndedError
from app.agent.history import get_account_history
from app.agent.identity import verify_identity
from app.agent.negotiation import (
    ProposalResult,
    TermsNotApprovedError,
    check_discount_request,
    propose_pay_in_full,
    propose_terms,
    record_promise,
)
from app.db.models import (
    ContactPrefs,
    Conversation,
    Debtor,
    Dispute,
    DisputeStatus,
    Event,
    Handoff,
    HandoffReason,
    Outcome,
    Promise,
    PromiseKind,
    PromiseStatus,
)
from app.db.seed import SeedPromise
from app.policy.config import load_rules
from tests.conftest import MakeDebtor

RULES = load_rules()
NOW = datetime(2026, 10, 24, 3, tzinfo=UTC)  # 12:00 JST
NAME = "Kenta Negotiate"
DOB = date(1984, 6, 1)
V = "v1"


async def _call(
    client: httpx.AsyncClient,
    make_debtor: MakeDebtor,
    *,
    verified: bool = True,
    engine: AsyncEngine | None = None,
    **debtor: Any,
) -> int:
    ref = await make_debtor(full_name=NAME, dob=DOB, **debtor)
    body = {
        "debtor_ref": ref,
        "sim_time": "2026-10-24T12:00:00+09:00",
        "target": "mobile",
        "agent_version": V,
    }
    cid = int((await client.post("/calls", json=body)).json()["conversation_id"])
    if verified:
        assert engine is not None
        async with engine.begin() as conn:
            await verify_identity(
                conn,
                conversation_id=cid,
                debtor_text=f"{NAME}, born 1984-06-01",
                now=NOW,
                rules=RULES,
                agent_version=V,
            )
    return cid


async def _propose(engine: AsyncEngine, cid: int, months: int, down: int) -> ProposalResult:
    async with engine.begin() as conn:
        return await propose_terms(
            conn,
            conversation_id=cid,
            months=months,
            down_payment_jpy=down,
            now=NOW,
            rules=RULES,
            agent_version=V,
        )


async def _promise(engine: AsyncEngine, cid: int, terms_id: int) -> str:
    async with engine.begin() as conn:
        return await record_promise(
            conn, conversation_id=cid, terms_id=terms_id, now=NOW, agent_version=V
        )


async def _scalar(engine: AsyncEngine, stmt: Any) -> Any:
    async with engine.connect() as conn:
        return await conn.scalar(stmt)


# --- History ------------------------------------------------------------------------------


async def test_history_blocked_before_verification(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(api_client, make_debtor, verified=False)
    async with app_engine.begin() as conn:
        result = await get_account_history(conn, conversation_id=cid, now=NOW, agent_version=V)
    assert result.allowed is False and result.facts is None
    decision = await _scalar(
        app_engine,
        select(Event.decision).where(Event.conversation_id == cid, Event.rule == "history_access"),
    )
    assert decision == "blocked"


async def test_history_after_verification_has_current_facts_only(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(
        api_client,
        make_debtor,
        engine=app_engine,
        promises=(
            SeedPromise(3, 10_000, 30_000, date(2026, 5, 1), PromiseStatus.SUPERSEDED),
            SeedPromise(3, 20_000, 30_000, date(2026, 8, 1), PromiseStatus.BROKEN),
        ),
        payments_jpy=(20_000,),
    )
    async with app_engine.begin() as conn:
        result = await get_account_history(conn, conversation_id=cid, now=NOW, agent_version=V)
    assert result.allowed and result.facts is not None
    facts = result.facts
    assert facts["balance_jpy"] == 100_000
    assert facts["open_dispute"] is False
    assert [p["status"] for p in facts["promises"]] == ["broken"]  # superseded excluded
    assert facts["has_broken_promise"] is True
    assert facts["payments_total_jpy"] == 20_000
    assert facts["last_payment_on"] is not None
    assert NAME not in str(facts) and "1984" not in str(facts)


# --- Terms gating -------------------------------------------------------------------------


async def test_terms_refused_before_verification(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(api_client, make_debtor, verified=False)
    result = await _propose(app_engine, cid, 3, 10_000)
    assert result.approved is False
    assert result.refusal_codes == ("identity_not_verified",)
    plan_events = await _scalar(
        app_engine,
        select(Event.id).where(Event.conversation_id == cid, Event.rule == "plan_terms"),
    )
    assert plan_events is None  # never reached the plan check


async def test_open_dispute_blocks_terms_and_requires_handoff(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(api_client, make_debtor, engine=app_engine, open_dispute="already paid")
    result = await _propose(app_engine, cid, 3, 10_000)
    assert result.approved is False
    assert result.handoff_required is True
    assert result.refusal_codes == ("open_dispute",)


async def test_out_of_policy_terms_refused_and_logged(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(api_client, make_debtor, engine=app_engine)
    result = await _propose(app_engine, cid, 7, 5_000)
    assert result.approved is False
    assert result.refusal_codes == ("months_out_of_range", "down_payment_too_low")
    assert result.terms_id is None


async def test_approved_terms_carry_code_built_schedule(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(api_client, make_debtor, engine=app_engine)
    result = await _propose(app_engine, cid, 3, 10_000)
    assert result.approved and result.terms_id is not None
    assert result.terms == (
        "A down payment of ¥10,000 by 2026-10-31, then 3 monthly installments of ¥30,000, "
        "from 2026-11-30 to 2027-01-31."
    )
    terms_input = await _scalar(app_engine, select(Event.input).where(Event.id == result.terms_id))
    assert sum(p["amount_jpy"] for p in terms_input["schedule"]) == 100_000


async def test_broken_promise_raises_down_payment_minimum(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(
        api_client,
        make_debtor,
        engine=app_engine,
        promises=(SeedPromise(3, 20_000, 30_000, date(2026, 8, 1), PromiseStatus.BROKEN),),
    )
    assert (await _propose(app_engine, cid, 3, 10_000)).refusal_codes == ("down_payment_too_low",)
    assert (await _propose(app_engine, cid, 3, 20_000)).approved is True


# --- Promises -----------------------------------------------------------------------------


async def test_record_promise_from_approved_terms(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(
        api_client,
        make_debtor,
        engine=app_engine,
        promises=(SeedPromise(4, 10_000, 22_500, date(2026, 9, 1), PromiseStatus.ACTIVE),),
    )
    proposal = await _propose(app_engine, cid, 3, 10_000)
    assert proposal.terms_id is not None and proposal.terms is not None
    say = await _promise(app_engine, cid, proposal.terms_id)
    assert say == scripts.promise_confirmed(proposal.terms)
    async with app_engine.connect() as conn:
        rows = (
            await conn.execute(
                select(Promise.status, Promise.kind, Promise.months, Promise.terms_event_id)
                .join(Conversation, Conversation.debtor_id == Promise.debtor_id)
                .where(Conversation.id == cid)
                .order_by(Promise.id)
            )
        ).all()
        outcome = await conn.scalar(select(Conversation.outcome).where(Conversation.id == cid))
    assert [r.status for r in rows] == [PromiseStatus.SUPERSEDED, PromiseStatus.ACTIVE]
    assert rows[1].kind is PromiseKind.INSTALLMENT_PLAN
    assert rows[1].months == 3
    assert rows[1].terms_event_id == proposal.terms_id
    assert outcome is Outcome.PROMISE


async def test_promise_only_from_latest_approved_terms(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(api_client, make_debtor, engine=app_engine)
    first = await _propose(app_engine, cid, 3, 10_000)
    refused = await _propose(app_engine, cid, 9, 0)
    second = await _propose(app_engine, cid, 6, 40_000)
    assert first.terms_id is not None and second.terms_id is not None
    with pytest.raises(TermsNotApprovedError):
        await _promise(app_engine, cid, first.terms_id)  # superseded by a newer approval
    with pytest.raises(TermsNotApprovedError):
        await _promise(app_engine, cid, 999_999_999)  # invented id
    assert refused.terms_id is None
    await _promise(app_engine, cid, second.terms_id)


async def test_terms_from_another_conversation_rejected(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    other = await _call(api_client, make_debtor, engine=app_engine)
    mine = await _call(api_client, make_debtor, engine=app_engine)
    foreign = await _propose(app_engine, other, 3, 10_000)
    assert foreign.terms_id is not None
    with pytest.raises(TermsNotApprovedError):
        await _promise(app_engine, mine, foreign.terms_id)


async def test_pay_in_full_promise(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(api_client, make_debtor, engine=app_engine)
    async with app_engine.begin() as conn:
        too_much = await propose_pay_in_full(
            conn,
            conversation_id=cid,
            amount_jpy=100_001,
            payment_date=date(2026, 10, 30),
            now=NOW,
            rules=RULES,
            agent_version=V,
        )
        ok = await propose_pay_in_full(
            conn,
            conversation_id=cid,
            amount_jpy=100_000,
            payment_date=date(2026, 10, 30),
            now=NOW,
            rules=RULES,
            agent_version=V,
        )
    assert too_much.refusal_codes == ("amount_exceeds_balance",)
    assert ok.approved and ok.terms == "One full payment of ¥100,000 by 2026-10-30."
    assert ok.terms_id is not None
    await _promise(app_engine, cid, ok.terms_id)
    kind = await _scalar(
        app_engine, select(Promise.kind).where(Promise.terms_event_id == ok.terms_id)
    )
    assert kind is PromiseKind.PAY_IN_FULL


async def test_discount_request_refused(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(api_client, make_debtor, engine=app_engine)
    async with app_engine.begin() as conn:
        result = await check_discount_request(
            conn, conversation_id=cid, offered_total_jpy=50_000, now=NOW, agent_version=V
        )
    assert result.approved is False
    assert result.refusal_codes == ("discount_not_allowed",)


# --- Actions ------------------------------------------------------------------------------


async def test_stop_contact_ends_call_and_blocks_future_dials(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(api_client, make_debtor, verified=False)
    async with app_engine.begin() as conn:
        say = await record_stop_contact(conn, conversation_id=cid, now=NOW, agent_version=V)
    assert say == scripts.STOP_CONTACT_END
    ref = await _scalar(
        app_engine,
        select(Debtor.external_ref).join(Conversation).where(Conversation.id == cid),
    )
    next_week = {
        "debtor_ref": ref,
        "sim_time": "2026-10-31T12:00:00+09:00",
        "target": "mobile",
        "agent_version": V,
    }
    body = (await api_client.post("/calls", json=next_week)).json()
    assert body["allowed"] is False
    assert body["decisions"][0]["codes"] == ["stop_contact_requested"]


async def test_no_workplace_flag_set_and_call_continues(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(api_client, make_debtor, engine=app_engine)
    async with app_engine.begin() as conn:
        await record_no_workplace_contact(conn, conversation_id=cid, now=NOW, agent_version=V)
    async with app_engine.connect() as conn:
        flag = await conn.scalar(
            select(ContactPrefs.no_workplace_contact)
            .join(Conversation, Conversation.debtor_id == ContactPrefs.debtor_id)
            .where(Conversation.id == cid)
        )
        outcome = await conn.scalar(select(Conversation.outcome).where(Conversation.id == cid))
    assert flag is True
    assert outcome is None


async def test_dispute_handoff_stores_dispute_and_blocks_next_call(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    cid = await _call(api_client, make_debtor, engine=app_engine)
    async with app_engine.begin() as conn:
        say = await request_handoff(
            conn,
            conversation_id=cid,
            reason=HandoffReason.DISPUTE,
            summary="Says this loan was never theirs.",
            now=NOW,
            agent_version=V,
        )
    assert say == scripts.HANDOFF_END
    async with app_engine.connect() as conn:
        debtor_id = await conn.scalar(select(Conversation.debtor_id).where(Conversation.id == cid))
        dispute = await conn.scalar(select(Dispute.status).where(Dispute.debtor_id == debtor_id))
        handoff = await conn.scalar(select(Handoff.reason).where(Handoff.conversation_id == cid))
        outcome = await conn.scalar(select(Conversation.outcome).where(Conversation.id == cid))
    assert dispute is DisputeStatus.OPEN
    assert handoff is HandoffReason.DISPUTE
    assert outcome is Outcome.HANDOFF
    with pytest.raises(ConversationEndedError):
        await _propose(app_engine, cid, 3, 10_000)
