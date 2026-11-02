"""Agent v5 payment ladder (rules-v8): the engine picks every offer; there is no discount rung.

Account used throughout: ¥100,000 owed, 10% minimum down (¥10,000), 3-6 months allowed.
"""

from datetime import date
from pathlib import Path

import pytest

from app.policy.config import PolicyRules, load_rules
from app.policy.engine import CustomerLimits, Rule, check_plan_terms
from app.policy.ladder import (
    Action,
    Offer,
    Rung,
    evaluate_proposal,
    ladder_offers,
    next_offer,
)
from tests.policy.conftest import jst

NOW = jst(2026, 10, 31, 12)
TODAY = date(2026, 10, 31)
BALANCE = 100_000
NONE = CustomerLimits()


def _offers(
    rules: PolicyRules, customer: CustomerLimits = NONE, broken: bool = False
) -> list[Offer]:
    return ladder_offers(
        today=TODAY, balance_jpy=BALANCE, has_broken_promise=broken, rules=rules, customer=customer
    )


def _next(
    rules: PolicyRules,
    offered: list[Rung],
    customer: CustomerLimits = NONE,
    **kw: int,
) -> tuple[Action, Offer | None, tuple[str, ...]]:
    result = next_offer(
        NOW,
        today=TODAY,
        balance_jpy=BALANCE,
        has_broken_promise=False,
        rules=rules,
        customer=customer,
        offered=offered,
        reason="first_offer",
        **kw,
    )
    (decision,) = result.decisions
    assert decision.rule is Rule.NEGOTIATION_LADDER
    assert decision.allowed is (result.action is Action.COUNTER)
    return result.action, result.offer, decision.codes


# --- The rungs that exist for an account ----------------------------------------------------


def test_global_rules_give_full_then_shortest_then_longest(rules: PolicyRules) -> None:
    assert _offers(rules) == [
        Offer(Rung.PAY_IN_FULL, 0, 0, 0, date(2026, 11, 14)),  # today + 14 days
        Offer(Rung.STANDARD_SPLIT, 3, 10_000, 30_000),
        Offer(Rung.EXTENDED_SPLIT, 6, 10_000, 15_000),
    ]


def test_customer_max_months_shortens_the_extended_plan(rules: PolicyRules) -> None:
    offers = _offers(rules, CustomerLimits(max_months=4))
    assert offers[-1] == Offer(Rung.EXTENDED_SPLIT, 4, 10_000, 22_500)


def test_customer_min_monthly_keeps_every_installment_at_or_above_it(rules: PolicyRules) -> None:
    offers = _offers(rules, CustomerLimits(min_monthly_jpy=20_000))
    assert [(o.rung, o.months, o.monthly_jpy) for o in offers[1:]] == [
        (Rung.STANDARD_SPLIT, 3, 30_000),
        (Rung.EXTENDED_SPLIT, 4, 22_500),  # 5 months would be ¥18,000 < ¥20,000
    ]


def test_min_monthly_above_every_plan_leaves_only_pay_in_full(rules: PolicyRules) -> None:
    offers = _offers(rules, CustomerLimits(min_monthly_jpy=40_000))
    assert [o.rung for o in offers] == [Rung.PAY_IN_FULL]


def test_one_valid_term_gives_no_separate_extended_plan(rules: PolicyRules) -> None:
    offers = _offers(rules, CustomerLimits(max_months=3))
    assert [o.rung for o in offers] == [Rung.PAY_IN_FULL, Rung.STANDARD_SPLIT]


def test_broken_promise_raises_the_down_payment_on_every_plan(rules: PolicyRules) -> None:
    plans = _offers(rules, broken=True)[1:]
    assert {o.down_payment_jpy for o in plans} == {20_000}
    assert [o.monthly_jpy for o in plans] == [26_666, 13_333]


def test_every_plan_collects_the_full_balance(rules: PolicyRules) -> None:
    """No discount rung: down payment + installments (last absorbs the rest) == balance."""
    for customer in (NONE, CustomerLimits(min_monthly_jpy=20_000), CustomerLimits(max_months=4)):
        for o in _offers(rules, customer)[1:]:
            remaining = BALANCE - o.down_payment_jpy
            last = remaining - o.monthly_jpy * (o.months - 1)
            assert o.down_payment_jpy + o.monthly_jpy * (o.months - 1) + last == BALANCE
            assert last >= o.monthly_jpy


# --- Next offer: down the ladder, never up, never repeated ----------------------------------


def test_first_offer_is_pay_in_full(rules: PolicyRules) -> None:
    action, offer, _ = _next(rules, [])
    assert action is Action.COUNTER and offer is not None and offer.rung is Rung.PAY_IN_FULL


def test_each_counter_moves_one_rung_down(rules: PolicyRules) -> None:
    _, standard, _ = _next(rules, [Rung.PAY_IN_FULL])
    _, extended, _ = _next(rules, [Rung.PAY_IN_FULL, Rung.STANDARD_SPLIT])
    assert standard is not None and standard.rung is Rung.STANDARD_SPLIT
    assert extended is not None and extended.rung is Rung.EXTENDED_SPLIT


def test_counter_limit_hands_off(rules: PolicyRules) -> None:
    assert rules.ladder is not None and rules.ladder.max_counter_offers == 2
    action, offer, codes = _next(
        rules, [Rung.PAY_IN_FULL, Rung.STANDARD_SPLIT, Rung.EXTENDED_SPLIT]
    )
    assert (action, offer, codes) == (Action.ESCALATE, None, ("counter_limit",))


def test_never_goes_back_up_the_ladder(rules: PolicyRules) -> None:
    action, _, codes = _next(rules, [Rung.EXTENDED_SPLIT])
    assert (action, codes) == (Action.ESCALATE, ("no_offer_left",))


def test_stated_monthly_limit_skips_to_a_plan_that_fits(rules: PolicyRules) -> None:
    action, offer, _ = _next(rules, [Rung.PAY_IN_FULL], affordable_monthly_jpy=20_000)
    assert action is Action.COUNTER and offer is not None
    assert (offer.rung, offer.monthly_jpy) == (Rung.EXTENDED_SPLIT, 15_000)


def test_stated_monthly_limit_below_every_plan_goes_to_a_person(rules: PolicyRules) -> None:
    action, offer, codes = _next(rules, [Rung.PAY_IN_FULL], affordable_monthly_jpy=10_000)
    assert (action, offer, codes) == (Action.ESCALATE, None, ("below_account_floor",))


def test_stated_down_payment_below_the_minimum_goes_to_a_person(rules: PolicyRules) -> None:
    action, _, codes = _next(rules, [Rung.PAY_IN_FULL], affordable_down_jpy=5_000)
    assert (action, codes) == (Action.ESCALATE, ("below_account_floor",))


def test_stated_limit_skips_pay_in_full(rules: PolicyRules) -> None:
    _, offer, _ = _next(rules, [], affordable_monthly_jpy=50_000)
    assert offer is not None and offer.rung is Rung.STANDARD_SPLIT


def test_no_plan_for_this_customer_after_pay_in_full_goes_to_a_person(
    rules: PolicyRules,
) -> None:
    action, _, codes = _next(rules, [Rung.PAY_IN_FULL], CustomerLimits(min_monthly_jpy=40_000))
    assert (action, codes) == (Action.ESCALATE, ("no_offer_left",))


def test_ladder_decision_records_what_the_engine_saw(rules: PolicyRules) -> None:
    result = next_offer(
        NOW,
        today=TODAY,
        balance_jpy=BALANCE,
        has_broken_promise=False,
        rules=rules,
        customer=CustomerLimits(max_months=4),
        offered=[Rung.PAY_IN_FULL],
        reason="cannot_pay_in_full",
    )
    inp = result.decisions[0].input
    assert inp["offer"]["rung"] == "standard_split"
    assert inp["customer_limits"] == {"min_monthly_jpy": None, "max_months": 4}
    assert (inp["reason"], inp["counters_used"], inp["rules_version"]) == (
        "cannot_pay_in_full",
        0,
        rules.version,
    )


def test_ladder_refuses_rules_without_a_ladder_block() -> None:
    v7 = load_rules(Path("tests/fixtures/rules-v7.yaml"))
    with pytest.raises(ValueError, match="no ladder block"):
        _next(v7, [])


# --- The customer's own plan ----------------------------------------------------------------


def _propose(
    rules: PolicyRules, months: int, down: int, customer: CustomerLimits = NONE, **kw: int
) -> tuple[Action, Offer | None, list[tuple[str, bool, tuple[str, ...]]]]:
    result = evaluate_proposal(
        NOW,
        today=TODAY,
        balance_jpy=BALANCE,
        months=months,
        down_payment_jpy=down,
        has_broken_promise=False,
        rules=rules,
        customer=customer,
        offered=[Rung.PAY_IN_FULL],
        **kw,
    )
    return (
        result.action,
        result.offer,
        [(d.rule.value, d.allowed, d.codes) for d in result.decisions],
    )


def test_customer_plan_inside_every_limit_is_accepted(rules: PolicyRules) -> None:
    action, offer, decisions = _propose(rules, 4, 10_000)
    assert action is Action.ACCEPT
    assert offer == Offer(Rung.CUSTOMER_PLAN, 4, 10_000, 22_500)
    assert decisions == [("plan_terms", True, ())]


def test_too_many_months_is_countered_with_the_lowest_monthly_plan(rules: PolicyRules) -> None:
    action, offer, decisions = _propose(rules, 12, 10_000)
    assert action is Action.COUNTER and offer is not None
    assert (offer.rung, offer.months, offer.monthly_jpy) == (Rung.EXTENDED_SPLIT, 6, 15_000)
    assert decisions == [
        ("plan_terms", False, ("months_out_of_range",)),
        ("negotiation_ladder", True, ()),
    ]


def test_above_customer_max_months_is_countered_within_their_limit(rules: PolicyRules) -> None:
    action, offer, decisions = _propose(rules, 6, 10_000, CustomerLimits(max_months=4))
    assert action is Action.COUNTER and offer is not None
    assert (offer.months, offer.monthly_jpy) == (4, 22_500)
    assert decisions[0] == ("plan_terms", False, ("above_customer_max_months",))


def test_below_customer_min_monthly_is_countered_at_or_above_it(rules: PolicyRules) -> None:
    action, offer, decisions = _propose(rules, 6, 10_000, CustomerLimits(min_monthly_jpy=20_000))
    assert action is Action.COUNTER and offer is not None
    assert offer.monthly_jpy >= 20_000
    assert decisions[0] == ("plan_terms", False, ("below_customer_min_monthly",))


def test_refused_plan_alone_is_never_a_reason_to_hand_off(rules: PolicyRules) -> None:
    """A customer's own plan only steers the pick; a handoff needs a stated limit."""
    action, offer, _ = _propose(rules, 6, 1_000)  # down payment too low, but nothing stated
    assert action is Action.COUNTER and offer is not None


def test_refused_plan_with_a_stated_down_limit_goes_to_a_person(rules: PolicyRules) -> None:
    action, offer, decisions = _propose(rules, 6, 0, affordable_down_jpy=0)
    assert (action, offer) == (Action.ESCALATE, None)
    assert decisions[-1] == ("negotiation_ladder", False, ("below_account_floor",))


# --- check_plan_terms with customer limits (v5 only) -----------------------------------------


@pytest.mark.parametrize(
    ("customer", "months", "allowed", "codes"),
    [
        (CustomerLimits(max_months=4), 4, True, ()),
        (CustomerLimits(max_months=4), 5, False, ("above_customer_max_months",)),
        (CustomerLimits(min_monthly_jpy=22_500), 4, True, ()),
        (CustomerLimits(min_monthly_jpy=22_501), 4, False, ("below_customer_min_monthly",)),
    ],
)
def test_plan_terms_with_customer_limits(
    rules: PolicyRules,
    customer: CustomerLimits,
    months: int,
    allowed: bool,
    codes: tuple[str, ...],
) -> None:
    d = check_plan_terms(
        NOW,
        balance_jpy=BALANCE,
        months=months,
        down_payment_jpy=10_000,
        has_broken_promise=False,
        rules=rules,
        customer=customer,
    )
    assert (d.allowed, d.codes) == (allowed, codes)
    assert d.input["customer_limits"] == customer.as_input()


def test_plan_terms_without_customer_limits_records_no_new_key(rules: PolicyRules) -> None:
    d = check_plan_terms(
        NOW,
        balance_jpy=BALANCE,
        months=6,
        down_payment_jpy=10_000,
        has_broken_promise=False,
        rules=rules,
    )
    assert d.allowed and "customer_limits" not in d.input


# --- V5-5: "can't pay it all" never gets pay in full next (calls 196, 202) ------------------


def _with_reason(rules: PolicyRules, reason: str) -> Offer | None:
    return next_offer(
        NOW,
        today=TODAY,
        balance_jpy=BALANCE,
        has_broken_promise=False,
        rules=rules,
        customer=NONE,
        offered=[],
        reason=reason,
    ).offer


@pytest.mark.parametrize(
    "reason", ["cannot_pay_in_full", "down_payment_too_high", "monthly_too_high", "other"]
)
def test_any_pushback_skips_the_pay_in_full_rung(rules: PolicyRules, reason: str) -> None:
    """V5-6: pay in full is only the opening offer, whatever reason the LLM picked."""
    offer = _with_reason(rules, reason)
    assert offer is not None and offer.rung is Rung.STANDARD_SPLIT


def test_the_opening_offer_is_pay_in_full(rules: PolicyRules) -> None:
    offer = _with_reason(rules, "first_offer")
    assert offer is not None and offer.rung is Rung.PAY_IN_FULL
