"""Two-phase terms gating.

1. propose_* asks the policy engine. An approved decision event *is* the approved terms:
   its id is the terms_id and its input holds the schedule built in code.
2. record_promise accepts only the latest approved terms in this conversation and reads
   every amount and date from that event, never from the LLM.
"""

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import exists, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from app.agent import scripts
from app.agent.conversation import OpenConversation, end_conversation, open_conversation, record
from app.agent.identity import is_verified
from app.db.models import (
    Debtor,
    Decision,
    Dispute,
    DisputeStatus,
    Event,
    EventType,
    Outcome,
    Promise,
    PromiseKind,
    PromiseStatus,
    Speaker,
    Turn,
)
from app.policy.callback import resolve_pay_in_full_date
from app.policy.config import PolicyRules
from app.policy.engine import (
    CustomerLimits,
    Rule,
    check_identity_required,
    check_no_discount,
    check_open_dispute,
    check_pay_in_full,
    check_plan_terms,
)
from app.policy.engine import Decision as PolicyDecision
from app.policy.ladder import Action, Offer, Rung, evaluate_proposal, next_offer
from app.policy.output_check import Figure, check_terms_presented
from app.policy.schedule import Payment, full_payment_schedule, plan_schedule

TERMS_RULES = (Rule.PLAN_TERMS.value, Rule.PAY_IN_FULL.value)


class TermsNotApprovedError(ValueError):
    """record_promise was called with terms that are not the latest approved ones."""


@dataclass(frozen=True)
class ProposalResult:
    approved: bool
    terms_id: int | None = None
    terms: str | None = None  # canonical text: the agent says exactly this
    refusal_codes: tuple[str, ...] = ()
    refusal_reason: str | None = None
    handoff_required: bool = False
    schedule: tuple[Payment, ...] = ()  # the approved schedule (v4: the LLM says it)


def _refusal(decision: PolicyDecision, *, handoff: bool = False) -> ProposalResult:
    return ProposalResult(
        approved=False,
        refusal_codes=decision.codes,
        refusal_reason=decision.reason,
        handoff_required=handoff,
    )


def _local_today(now: datetime, tz: str) -> date:
    return now.astimezone(ZoneInfo(tz)).date()


def _dump(schedule: list[Payment]) -> list[dict[str, Any]]:
    return [
        {"due": p.due.isoformat(), "amount_jpy": p.amount_jpy, "kind": p.kind} for p in schedule
    ]


async def terms_schedule(conn: AsyncConnection, terms_id: int) -> list[Payment]:
    """The payment schedule stored with approved terms (code-built, never from the LLM)."""
    terms_input = await conn.scalar(select(Event.input).where(Event.id == terms_id))
    assert terms_input is not None
    return _load(terms_input["schedule"])


def _load(raw: list[dict[str, Any]]) -> list[Payment]:
    return [
        Payment(date.fromisoformat(p["due"]), int(p["amount_jpy"]), str(p["kind"])) for p in raw
    ]


async def _has_open_dispute(conn: AsyncConnection, debtor_id: int) -> bool:
    return bool(
        await conn.scalar(
            select(
                exists().where(Dispute.debtor_id == debtor_id, Dispute.status == DisputeStatus.OPEN)
            )
        )
    )


async def _has_broken_promise(conn: AsyncConnection, debtor_id: int) -> bool:
    return bool(
        await conn.scalar(
            select(
                exists().where(
                    Promise.debtor_id == debtor_id, Promise.status == PromiseStatus.BROKEN
                )
            )
        )
    )


async def _gate(
    conn: AsyncConnection, convo: OpenConversation, now: datetime, action: str, agent_version: str
) -> ProposalResult | None:
    """Identity first, then open dispute. Returns a refusal, or None when allowed."""
    identity = check_identity_required(
        now, identity_verified=await is_verified(conn, convo.conversation_id), action=action
    )
    await record(conn, convo, identity, agent_version)
    if not identity.allowed:
        return _refusal(identity)
    dispute = check_open_dispute(now, await _has_open_dispute(conn, convo.debtor_id))
    await record(conn, convo, dispute, agent_version)
    if not dispute.allowed:
        return _refusal(dispute, handoff=True)
    return None


async def propose_terms(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    months: int,
    down_payment_jpy: int,
    now: datetime,
    rules: PolicyRules,
    agent_version: str,
) -> ProposalResult:
    convo = await open_conversation(conn, conversation_id)
    if refusal := await _gate(conn, convo, now, "propose_terms", agent_version):
        return refusal
    decision = check_plan_terms(
        now,
        balance_jpy=convo.balance_jpy,
        months=months,
        down_payment_jpy=down_payment_jpy,
        has_broken_promise=await _has_broken_promise(conn, convo.debtor_id),
        rules=rules,
    )
    if not decision.allowed:
        await record(conn, convo, decision, agent_version)
        return _refusal(decision)
    down_due = _local_today(now, convo.timezone) + timedelta(days=rules.plan.down_payment_due_days)
    schedule = plan_schedule(
        balance_jpy=convo.balance_jpy,
        months=months,
        down_payment_jpy=down_payment_jpy,
        down_payment_due=down_due,
    )
    decision = replace(
        decision,
        input=decision.input
        | {"kind": PromiseKind.INSTALLMENT_PLAN.value, "schedule": _dump(schedule)},
    )
    terms_id = await record(conn, convo, decision, agent_version)
    return ProposalResult(
        True, terms_id=terms_id, terms=scripts.terms_text(schedule), schedule=tuple(schedule)
    )


async def propose_pay_in_full(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    amount_jpy: int,
    payment_date: date | None = None,
    day_phrase: str | None = None,
    now: datetime,
    rules: PolicyRules,
    agent_version: str,
) -> ProposalResult:
    """payment_date: v1-v4 (the LLM's date). day_phrase: v5 (V5-5), the customer's words for
    the day; code resolves them against what the customer said in this call."""
    convo = await open_conversation(conn, conversation_id)
    if refusal := await _gate(conn, convo, now, "propose_pay_in_full", agent_version):
        return refusal
    if day_phrase is not None:
        said = list(
            await conn.scalars(
                select(Turn.text).where(
                    Turn.conversation_id == conversation_id, Turn.speaker == Speaker.DEBTOR
                )
            )
        )
        payment_date, blocked = resolve_pay_in_full_date(
            now, debtor_tz=convo.timezone, day_phrase=day_phrase, said=said, rules=rules
        )
        if blocked is not None:
            await record(conn, convo, blocked, agent_version)
            return _refusal(blocked)
    if payment_date is None:
        raise ValueError("propose_pay_in_full needs payment_date or day_phrase")
    decision = check_pay_in_full(
        now,
        debtor_tz=convo.timezone,
        balance_jpy=convo.balance_jpy,
        amount_jpy=amount_jpy,
        payment_date=payment_date,
        rules=rules,
    )
    if not decision.allowed:
        await record(conn, convo, decision, agent_version)
        return _refusal(decision)
    schedule = full_payment_schedule(amount_jpy=amount_jpy, due=payment_date)
    decision = replace(
        decision,
        input=decision.input | {"kind": PromiseKind.PAY_IN_FULL.value, "schedule": _dump(schedule)},
    )
    terms_id = await record(conn, convo, decision, agent_version)
    return ProposalResult(
        True, terms_id=terms_id, terms=scripts.terms_text(schedule), schedule=tuple(schedule)
    )


async def check_discount_request(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    offered_total_jpy: int,
    now: datetime,
    agent_version: str,
    reject_zero_offer: bool = False,
) -> ProposalResult:
    """Debtor asks to pay less ("your manager approved 50% off"). Always audited."""
    convo = await open_conversation(conn, conversation_id)
    if refusal := await _gate(conn, convo, now, "discount_request", agent_version):
        return refusal
    decision = check_no_discount(
        now,
        balance_jpy=convo.balance_jpy,
        offered_total_jpy=offered_total_jpy,
        reject_zero_offer=reject_zero_offer,
    )
    await record(conn, convo, decision, agent_version)
    if not decision.allowed:
        return _refusal(decision)
    return ProposalResult(approved=True)  # full balance: nothing to refuse


async def record_promise(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    terms_id: int,
    now: datetime,
    agent_version: str,
) -> str:
    convo = await open_conversation(conn, conversation_id)
    latest = await conn.scalar(
        select(func.max(Event.id)).where(
            Event.conversation_id == conversation_id,
            Event.type == EventType.POLICY_DECISION,
            Event.rule.in_(TERMS_RULES),
            Event.decision == Decision.ALLOWED,
        )
    )
    if latest is None or latest != terms_id:
        raise TermsNotApprovedError(f"terms {terms_id} are not the latest approved terms")
    terms_input = await conn.scalar(select(Event.input).where(Event.id == terms_id))
    assert terms_input is not None
    schedule = _load(terms_input["schedule"])
    kind = PromiseKind(terms_input["kind"])
    first = schedule[0] if kind is PromiseKind.PAY_IN_FULL else schedule[1]

    # A new promise supersedes any active one; only current facts reach the LLM later.
    await conn.execute(
        update(Promise)
        .where(Promise.debtor_id == convo.debtor_id, Promise.status == PromiseStatus.ACTIVE)
        .values(status=PromiseStatus.SUPERSEDED, status_changed_at=now)
    )
    await conn.execute(
        insert(Promise).values(
            debtor_id=convo.debtor_id,
            conversation_id=conversation_id,
            kind=kind,
            months=len(schedule) - 1 if kind is PromiseKind.INSTALLMENT_PLAN else 1,
            down_payment_jpy=schedule[0].amount_jpy if kind is PromiseKind.INSTALLMENT_PLAN else 0,
            installment_jpy=first.amount_jpy,
            first_due_date=first.due,
            status=PromiseStatus.ACTIVE,
            status_changed_at=now,
            terms_event_id=terms_id,
        )
    )
    await end_conversation(
        conn,
        convo,
        Outcome.PROMISE,
        reason="promise recorded",
        now=now,
        agent_version=agent_version,
    )
    return scripts.promise_confirmed(scripts.terms_text(schedule))


def required_figures(schedule: list[Payment]) -> list[Figure]:
    """What the person must have heard before a promise (agent v4)."""
    if len(schedule) == 1:
        return [Figure("amount", schedule[0].amount_jpy), Figure("date", schedule[0].due)]
    return [
        Figure("amount", schedule[0].amount_jpy),
        Figure("amount", schedule[1].amount_jpy),
        Figure("count", len(schedule) - 1),
    ]


async def terms_were_said(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    terms_id: int,
    said: list[str],
    now: datetime,
    rules: PolicyRules,
    agent_version: str,
) -> bool:
    """Agent v4 (fixes.llm_presents_terms): every required figure of the approved terms was
    said since they were approved. Audited either way. Unknown terms_id: True here, and
    record_promise then refuses it as not the latest approved terms."""
    terms_input = await conn.scalar(
        select(Event.input).where(
            Event.id == terms_id,
            Event.conversation_id == conversation_id,
            Event.rule.in_(TERMS_RULES),
            Event.decision == Decision.ALLOWED,
        )
    )
    if terms_input is None:
        return True
    convo = await open_conversation(conn, conversation_id)
    decision = check_terms_presented(
        now,
        terms_id=terms_id,
        required=required_figures(_load(terms_input["schedule"])),
        said=said,
        rules=rules,
    )
    await record(conn, convo, decision, agent_version)
    return decision.allowed


# --- Agent v5 (fixes.payment_ladder, rules-v8): the engine picks every offer ------------------


@dataclass(frozen=True)
class LadderOutcome:
    """What the engine decided. proposal: approved terms (accept or counter) or a gate refusal
    (identity, open dispute). escalate: the code of a ladder handoff, else None."""

    action: Action | None
    proposal: ProposalResult | None = None
    escalate: str | None = None
    rung: Rung | None = None
    refused_codes: tuple[str, ...] = ()  # the customer's own plan, when it was not allowed


async def _customer_limits(conn: AsyncConnection, debtor_id: int) -> CustomerLimits:
    row = (
        await conn.execute(
            select(Debtor.min_monthly_jpy, Debtor.max_months).where(Debtor.id == debtor_id)
        )
    ).one()
    return CustomerLimits(min_monthly_jpy=row.min_monthly_jpy, max_months=row.max_months)


async def offered_rungs(conn: AsyncConnection, conversation_id: int) -> list[Rung]:
    """Rungs the engine already offered in this call, from its own audit events (not memory)."""
    rows = await conn.scalars(
        select(Event.input["offer"]["rung"].astext)
        .where(
            Event.conversation_id == conversation_id,
            Event.rule == Rule.NEGOTIATION_LADDER.value,
            Event.decision == Decision.ALLOWED,
        )
        .order_by(Event.id)
    )
    return [Rung(r) for r in rows if r]


async def _approve(
    conn: AsyncConnection,
    convo: OpenConversation,
    offer: Offer,
    *,
    now: datetime,
    rules: PolicyRules,
    agent_version: str,
    customer: CustomerLimits,
    has_broken_promise: bool,
) -> ProposalResult:
    """An engine offer (or an accepted customer plan) becomes approved terms: the same kind
    of decision event propose_terms records, so record_promise works unchanged."""
    if offer.rung is Rung.PAY_IN_FULL:
        assert offer.payment_date is not None
        decision = check_pay_in_full(
            now,
            debtor_tz=convo.timezone,
            balance_jpy=convo.balance_jpy,
            amount_jpy=convo.balance_jpy,
            payment_date=offer.payment_date,
            rules=rules,
        )
        schedule = full_payment_schedule(amount_jpy=convo.balance_jpy, due=offer.payment_date)
        kind = PromiseKind.PAY_IN_FULL
    else:
        decision = check_plan_terms(
            now,
            balance_jpy=convo.balance_jpy,
            months=offer.months,
            down_payment_jpy=offer.down_payment_jpy,
            has_broken_promise=has_broken_promise,
            rules=rules,
            customer=customer,
        )
        down_due = _local_today(now, convo.timezone) + timedelta(
            days=rules.plan.down_payment_due_days
        )
        schedule = plan_schedule(
            balance_jpy=convo.balance_jpy,
            months=offer.months,
            down_payment_jpy=offer.down_payment_jpy,
            down_payment_due=down_due,
        )
        kind = PromiseKind.INSTALLMENT_PLAN
    if not decision.allowed:  # never expected: the ladder only builds allowed offers
        await record(conn, convo, decision, agent_version)
        return _refusal(decision)
    decision = replace(
        decision,
        input=decision.input
        | {"kind": kind.value, "schedule": _dump(schedule), "ladder_rung": offer.rung.value},
    )
    terms_id = await record(conn, convo, decision, agent_version)
    return ProposalResult(
        True, terms_id=terms_id, terms=scripts.terms_text(schedule), schedule=tuple(schedule)
    )


async def ladder_next_offer(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    reason: str,
    affordable_monthly_jpy: int | None,
    affordable_down_jpy: int | None,
    now: datetime,
    rules: PolicyRules,
    agent_version: str,
) -> LadderOutcome:
    """v5 next_offer: the engine's next rung, approved as terms, or a ladder handoff."""
    convo = await open_conversation(conn, conversation_id)
    if refusal := await _gate(conn, convo, now, "next_offer", agent_version):
        return LadderOutcome(None, proposal=refusal)
    customer = await _customer_limits(conn, convo.debtor_id)
    broken = await _has_broken_promise(conn, convo.debtor_id)
    result = next_offer(
        now,
        today=_local_today(now, convo.timezone),
        balance_jpy=convo.balance_jpy,
        has_broken_promise=broken,
        rules=rules,
        customer=customer,
        offered=await offered_rungs(conn, conversation_id),
        reason=reason,
        affordable_monthly_jpy=affordable_monthly_jpy,
        affordable_down_jpy=affordable_down_jpy,
    )
    for d in result.decisions:
        await record(conn, convo, d, agent_version)
    if result.action is Action.ESCALATE or result.offer is None:
        return LadderOutcome(Action.ESCALATE, escalate=result.decisions[-1].codes[0])
    proposal = await _approve(
        conn,
        convo,
        result.offer,
        now=now,
        rules=rules,
        agent_version=agent_version,
        customer=customer,
        has_broken_promise=broken,
    )
    return LadderOutcome(Action.COUNTER, proposal=proposal, rung=result.offer.rung)


async def ladder_propose_terms(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    months: int,
    down_payment_jpy: int,
    now: datetime,
    rules: PolicyRules,
    agent_version: str,
) -> LadderOutcome:
    """v5 propose_terms: the customer's own plan is accepted, countered by the engine, or
    handed off. Never a bare refusal."""
    convo = await open_conversation(conn, conversation_id)
    if refusal := await _gate(conn, convo, now, "propose_terms", agent_version):
        return LadderOutcome(None, proposal=refusal)
    customer = await _customer_limits(conn, convo.debtor_id)
    broken = await _has_broken_promise(conn, convo.debtor_id)
    result = evaluate_proposal(
        now,
        today=_local_today(now, convo.timezone),
        balance_jpy=convo.balance_jpy,
        months=months,
        down_payment_jpy=down_payment_jpy,
        has_broken_promise=broken,
        rules=rules,
        customer=customer,
        offered=await offered_rungs(conn, conversation_id),
    )
    terms, *ladder = result.decisions
    if result.action is Action.ACCEPT and result.offer is not None:
        # The plan_terms decision above is allowed; _approve records it with its schedule.
        proposal = await _approve(
            conn,
            convo,
            result.offer,
            now=now,
            rules=rules,
            agent_version=agent_version,
            customer=customer,
            has_broken_promise=broken,
        )
        return LadderOutcome(Action.ACCEPT, proposal=proposal, rung=Rung.CUSTOMER_PLAN)
    for d in (terms, *ladder):
        await record(conn, convo, d, agent_version)
    if result.action is Action.ESCALATE or result.offer is None:
        return LadderOutcome(
            Action.ESCALATE, escalate=ladder[-1].codes[0], refused_codes=terms.codes
        )
    proposal = await _approve(
        conn,
        convo,
        result.offer,
        now=now,
        rules=rules,
        agent_version=agent_version,
        customer=customer,
        has_broken_promise=broken,
    )
    return LadderOutcome(
        Action.COUNTER, proposal=proposal, rung=result.offer.rung, refused_codes=terms.codes
    )
