"""Agent v5 callbacks (rules-v8): only a day the customer named, tomorrow .. +30 days."""

from datetime import date
from pathlib import Path

import pytest

from app.policy.callback import check_callback, customer_dates, resolve_pay_in_full_date
from app.policy.config import PolicyRules, load_rules
from app.policy.engine import Rule
from tests.policy.conftest import jst

NOW = jst(2026, 10, 31, 12)  # Saturday 31 October 2026, noon JST
TODAY = date(2026, 10, 31)


@pytest.mark.parametrize(
    ("said", "expected"),
    [
        ("I get paid Friday", {date(2026, 11, 6)}),
        ("call me on saturday", {date(2026, 11, 7)}),  # today is Saturday: the next one
        ("maybe next Friday", {date(2026, 11, 6), date(2026, 11, 13)}),
        ("payday is the 25th", {date(2026, 11, 25)}),
        ("on the 31st", set()),  # today is the 31st and November has none: nothing to book
        ("tomorrow works", {date(2026, 11, 1)}),
        ("in two weeks", {date(2026, 11, 14)}),
        ("in 3 days", {date(2026, 11, 3)}),
        ("end of the month", {date(2026, 11, 30)}),  # today is the last day of October
        ("November 10th, 2026 please", {date(2026, 11, 10)}),
    ],
)
def test_customer_dates_from_their_own_words(said: str, expected: set[date]) -> None:
    assert set(customer_dates([said], TODAY)) == expected


def test_next_week_and_next_month_cover_the_whole_period() -> None:
    week = set(customer_dates(["sometime next week"], TODAY))
    assert week == {date(2026, 11, d) for d in range(2, 9)}  # Monday 2 .. Sunday 8 November
    month = set(customer_dates(["next month is better"], TODAY))
    assert len(month) == 30 and min(month) == date(2026, 11, 1)


def test_numbers_that_are_not_days_are_not_dates() -> None:
    assert customer_dates(["I can pay 5000 yen, maybe 25 or 30 thousand"], TODAY) == {}


def _check(rules: PolicyRules, requested: date, said: list[str]) -> tuple[bool, tuple[str, ...]]:
    d = check_callback(NOW, debtor_tz="Asia/Tokyo", requested=requested, said=said, rules=rules)
    assert d.rule is Rule.CALLBACK_REQUEST
    return d.allowed, d.codes


def test_a_day_the_customer_named_is_allowed(rules: PolicyRules) -> None:
    assert _check(rules, date(2026, 11, 6), ["I get paid on Friday."]) == (True, ())


def test_a_day_the_customer_never_named_is_blocked(rules: PolicyRules) -> None:
    assert _check(rules, date(2026, 11, 9), ["I get paid on Friday."]) == (
        False,
        ("date_not_said",),
    )


def test_today_is_too_soon(rules: PolicyRules) -> None:
    allowed, codes = _check(rules, TODAY, ["call me today"])
    assert not allowed and "too_soon" in codes


def test_more_than_30_days_ahead_is_too_far(rules: PolicyRules) -> None:
    allowed, codes = _check(rules, date(2026, 12, 25), ["call me on December 25th, 2026"])
    assert (allowed, codes) == (False, ("too_far",))


def test_the_last_allowed_day_is_allowed(rules: PolicyRules) -> None:
    assert _check(rules, date(2026, 11, 30), ["November 30th, 2026"]) == (True, ())


def test_the_debtors_own_timezone_decides_today(rules: PolicyRules) -> None:
    """Noon JST on the 31st is still the 30th in New York: 'tomorrow' there is the 31st."""
    d = check_callback(
        NOW,
        debtor_tz="America/New_York",
        requested=date(2026, 10, 31),
        said=["tomorrow"],
        rules=rules,
    )
    assert d.allowed and d.input["debtor_local_today"] == "2026-10-30"


def test_the_decision_records_the_customers_phrase(rules: PolicyRules) -> None:
    d = check_callback(
        NOW, debtor_tz="Asia/Tokyo", requested=date(2026, 11, 6), said=["paid Friday"], rules=rules
    )
    assert d.input["customer_phrase"] == "friday"
    assert d.input["max_days_ahead"] == 30


def test_rules_without_a_callback_block_are_refused() -> None:
    v7 = load_rules(Path("tests/fixtures/rules-v7.yaml"))
    with pytest.raises(ValueError, match="no callback block"):
        _check(v7, date(2026, 11, 6), ["Friday"])


# --- V5-5: a named month is kept (call 193: "January 20th" was booked as November 20) -------


@pytest.mark.parametrize(
    ("said", "expected"),
    [
        ("call me on January 20th", {date(2027, 1, 20)}),
        ("the 20th of January", {date(2027, 1, 20)}),
        ("Jan 20th, 2027", {date(2027, 1, 20)}),
        ("November 20th works", {date(2026, 11, 20)}),
        ("I can pay today", {TODAY}),
    ],
)
def test_a_named_month_is_never_dropped(said: str, expected: set[date]) -> None:
    assert set(customer_dates([said], TODAY)) == expected


def test_a_named_month_past_the_limit_is_blocked_not_moved(rules: PolicyRules) -> None:
    said = ["Can you call me back on January 20th?"]
    assert _check(rules, date(2027, 1, 20), said) == (False, ("too_far",))
    assert _check(rules, date(2026, 11, 20), said) == (False, ("date_not_said",))
    assert _check(rules, date(2026, 11, 20), ["November 20th please"]) == (True, ())


# --- V5-5: pay-in-full dates come from the customer's words (call 198) ----------------------


def test_pay_in_full_day_the_customer_said_is_resolved_in_code(rules: PolicyRules) -> None:
    when, blocked = resolve_pay_in_full_date(
        NOW,
        debtor_tz="Asia/Tokyo",
        day_phrase="Friday",
        said=["I can pay it all Friday"],
        rules=rules,
    )
    assert (when, blocked) == (date(2026, 11, 6), None)


@pytest.mark.parametrize(
    ("phrase", "said"),
    [
        ("Friday", ["I can pay it all soon"]),  # never said
        ("2026-10-27", ["I can pay it all Friday"]),  # a date the LLM worked out (in the past)
        ("", ["I can pay it all Friday"]),
    ],
)
def test_pay_in_full_day_not_said_is_blocked(
    rules: PolicyRules, phrase: str, said: list[str]
) -> None:
    when, blocked = resolve_pay_in_full_date(
        NOW, debtor_tz="Asia/Tokyo", day_phrase=phrase, said=said, rules=rules
    )
    assert when is None and blocked is not None
    assert (blocked.rule, blocked.allowed, blocked.codes) == (
        Rule.PAY_IN_FULL,
        False,
        ("date_not_said",),
    )
