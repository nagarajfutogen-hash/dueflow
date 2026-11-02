"""Agent v5 (rules-v8): the payment ladder. Pure: the engine, not the LLM, picks the next offer.

Rungs, in order: pay in full -> standard plan (shortest term) -> extended plan (the longest term
the customer's limits allow, so the lowest monthly amount) -> handoff. The ladder only moves
down, never repeats a rung, and stops after `ladder.max_counter_offers` counter-offers.

There is no discount rung: every offer collects the full balance (PLANNING §6, owner decision 1A).
"""

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from app.policy.config import PolicyRules
from app.policy.engine import (
    CustomerLimits,
    Decision,
    Rule,
    check_plan_terms,
    min_down_payment,
)


class Rung(enum.StrEnum):
    PAY_IN_FULL = "pay_in_full"
    STANDARD_SPLIT = "standard_split"
    EXTENDED_SPLIT = "extended_split"
    CUSTOMER_PLAN = "customer_plan"  # the customer's own proposal, accepted as it is


LADDER = (Rung.PAY_IN_FULL, Rung.STANDARD_SPLIT, Rung.EXTENDED_SPLIT)


class Action(enum.StrEnum):
    ACCEPT = "accept"
    COUNTER = "counter"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class Offer:
    rung: Rung
    months: int  # monthly installments after the down payment; 0 = one full payment
    down_payment_jpy: int
    monthly_jpy: int  # the first monthly installment (the last absorbs any remainder)
    payment_date: date | None = None  # pay in full only

    def as_input(self) -> dict[str, Any]:
        return {
            "rung": self.rung.value,
            "months": self.months,
            "down_payment_jpy": self.down_payment_jpy,
            "monthly_jpy": self.monthly_jpy,
            "payment_date": self.payment_date.isoformat() if self.payment_date else None,
        }


@dataclass(frozen=True)
class LadderResult:
    action: Action
    offer: Offer | None
    # Every decision taken, in order, for the caller to record (allowed and blocked).
    decisions: tuple[Decision, ...]


def _require(rules: PolicyRules) -> int:
    if rules.ladder is None:
        raise ValueError(f"{rules.version} has no ladder block (rules-v8 or later)")
    return rules.ladder.max_counter_offers


def ladder_offers(
    *,
    today: date,
    balance_jpy: int,
    has_broken_promise: bool,
    rules: PolicyRules,
    customer: CustomerLimits,
) -> list[Offer]:
    """Every rung that exists for this account, in ladder order. `today` is debtor-local."""
    plan = rules.plan
    offers = [
        Offer(
            Rung.PAY_IN_FULL,
            months=0,
            down_payment_jpy=0,
            monthly_jpy=0,
            payment_date=today + timedelta(days=rules.pay_in_full.max_days_until_payment),
        )
    ]
    down = min_down_payment(balance_jpy, has_broken_promise=has_broken_promise, rules=rules)
    remaining = balance_jpy - down
    if remaining <= 0:
        return offers
    max_months = min(plan.max_months, customer.max_months or plan.max_months)
    floor = max(1, customer.min_monthly_jpy or 1)
    valid = [m for m in range(plan.min_months, max_months + 1) if remaining // m >= floor]
    if valid:
        shortest, longest = min(valid), max(valid)
        offers.append(Offer(Rung.STANDARD_SPLIT, shortest, down, remaining // shortest))
        if longest > shortest:
            offers.append(Offer(Rung.EXTENDED_SPLIT, longest, down, remaining // longest))
    return offers


def _affordable(offer: Offer, balance_jpy: int, monthly: int | None, down: int | None) -> bool:
    if offer.rung is Rung.PAY_IN_FULL:
        return (monthly is None or monthly >= balance_jpy) and (down is None or down >= balance_jpy)
    return (monthly is None or offer.monthly_jpy <= monthly) and (
        down is None or offer.down_payment_jpy <= down
    )


def next_offer(
    now: datetime,
    *,
    today: date,
    balance_jpy: int,
    has_broken_promise: bool,
    rules: PolicyRules,
    customer: CustomerLimits,
    offered: Sequence[Rung],
    reason: str,
    affordable_monthly_jpy: int | None = None,
    affordable_down_jpy: int | None = None,
    preferred_monthly_jpy: int | None = None,
) -> LadderResult:
    """The next offer down the ladder, or a handoff.

    offered: rungs already offered in this call (the first offer is not a counter-offer).
    reason: why the customer needs another option, as the agent classified it (audited).
    affordable_*: what the customer said they can pay, if they said it; an offer above it is
    skipped. If every remaining offer is above it, the call goes to a person
    (below_account_floor): the engine never lowers the amount owed.
    preferred_monthly_jpy: a monthly amount the customer asked for but did not say is their
    limit (a refused plan of their own). It only steers the pick: the first plan at or below
    it, else the plan with the lowest monthly amount. Never a reason to hand off.
    """
    max_counters = _require(rules)
    offers = ladder_offers(
        today=today,
        balance_jpy=balance_jpy,
        has_broken_promise=has_broken_promise,
        rules=rules,
        customer=customer,
    )
    lowest_offered = max((LADDER.index(r) for r in offered if r in LADDER), default=-1)
    remaining = [o for o in offers if LADDER.index(o.rung) > lowest_offered]
    if reason != "first_offer":
        # V5-6: pay in full is only ever the opening offer. Any other reason means the person
        # already pushed back (the LLM mislabelled "can't pay it all" in calls 210, 212).
        remaining = [o for o in remaining if o.rung is not Rung.PAY_IN_FULL]
    fits = [
        o
        for o in remaining
        if _affordable(o, balance_jpy, affordable_monthly_jpy, affordable_down_jpy)
    ]
    counters_used = max(0, len(offered) - 1)
    inp: dict[str, Any] = {
        "balance_jpy": balance_jpy,
        "has_broken_promise": has_broken_promise,
        "customer_limits": customer.as_input(),
        "offered": [r.value for r in offered],
        "counters_used": counters_used,
        "max_counter_offers": max_counters,
        "reason": reason,
        "affordable_monthly_jpy": affordable_monthly_jpy,
        "affordable_down_jpy": affordable_down_jpy,
        "preferred_monthly_jpy": preferred_monthly_jpy,
        "rules_version": rules.version,
    }

    def escalate(code: str, why: str) -> LadderResult:
        inp["action"] = Action.ESCALATE.value
        decision = Decision(
            rule=Rule.NEGOTIATION_LADDER,
            allowed=False,
            reason=why,
            at=now,
            input=inp,
            codes=(code,),
        )
        return LadderResult(Action.ESCALATE, None, (decision,))

    if offered and counters_used >= max_counters:
        return escalate("counter_limit", f"{counters_used} counter-offers made; limit reached")
    if not fits:
        if remaining:
            return escalate(
                "below_account_floor",
                "every remaining offer is above what the customer can pay",
            )
        return escalate("no_offer_left", "every allowed offer was already made")
    offer = fits[0]
    plans = [o for o in fits if o.rung is not Rung.PAY_IN_FULL]
    if preferred_monthly_jpy is not None and plans:
        within = [o for o in plans if o.monthly_jpy <= preferred_monthly_jpy]
        offer = within[0] if within else min(plans, key=lambda o: o.monthly_jpy)
    inp |= {"action": Action.COUNTER.value, "offer": offer.as_input()}
    decision = Decision(
        rule=Rule.NEGOTIATION_LADDER,
        allowed=True,
        reason=f"next offer: {offer.rung.value}",
        at=now,
        input=inp,
    )
    return LadderResult(Action.COUNTER, offer, (decision,))


def evaluate_proposal(
    now: datetime,
    *,
    today: date,
    balance_jpy: int,
    months: int,
    down_payment_jpy: int,
    has_broken_promise: bool,
    rules: PolicyRules,
    customer: CustomerLimits,
    offered: Sequence[Rung],
    affordable_down_jpy: int | None = None,
) -> LadderResult:
    """The customer's own plan: accept it if it fits every rule and the customer's limits;
    otherwise counter with the next rung they can afford, or hand off."""
    _require(rules)
    terms = check_plan_terms(
        now,
        balance_jpy=balance_jpy,
        months=months,
        down_payment_jpy=down_payment_jpy,
        has_broken_promise=has_broken_promise,
        rules=rules,
        customer=customer,
    )
    if terms.allowed:
        monthly = (balance_jpy - down_payment_jpy) // months
        offer = Offer(Rung.CUSTOMER_PLAN, months, down_payment_jpy, monthly)
        return LadderResult(Action.ACCEPT, offer, (terms,))
    # What they asked for per month steers the pick; a hard limit only if they said it.
    wanted_monthly = (balance_jpy - down_payment_jpy) // months if months > 0 else None
    result = next_offer(
        now,
        today=today,
        balance_jpy=balance_jpy,
        has_broken_promise=has_broken_promise,
        rules=rules,
        customer=customer,
        offered=offered,
        reason="customer_plan_not_allowed",
        affordable_down_jpy=affordable_down_jpy,
        preferred_monthly_jpy=wanted_monthly,
    )
    return LadderResult(result.action, result.offer, (terms, *result.decisions))
