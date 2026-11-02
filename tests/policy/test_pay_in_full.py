from datetime import UTC, date, datetime

import pytest

from app.policy.config import PolicyRules
from app.policy.engine import Rule, check_pay_in_full, check_plan_terms
from tests.policy.conftest import jst

NOW = jst(2026, 10, 24, 12)
TODAY = date(2026, 10, 24)
BALANCE = 100_000


def _pay(
    rules: PolicyRules,
    amount: int = BALANCE,
    on: date = TODAY,
    balance: int = BALANCE,
    now: datetime = NOW,
    tz: str = "Asia/Tokyo",
) -> tuple[bool, tuple[str, ...]]:
    d = check_pay_in_full(
        now, debtor_tz=tz, balance_jpy=balance, amount_jpy=amount, payment_date=on, rules=rules
    )
    assert d.rule is Rule.PAY_IN_FULL
    return d.allowed, d.codes


@pytest.mark.parametrize(
    "on",
    [date(2026, 10, 24), date(2026, 10, 25), date(2026, 11, 7)],
    ids=["today", "tomorrow", "last_allowed_day"],
)
def test_pay_in_full_allowed(rules: PolicyRules, on: date) -> None:
    assert _pay(rules, on=on) == (True, ())


def test_full_balance_as_plan_down_payment_is_redirected(rules: PolicyRules) -> None:
    # Paying everything is not an installment plan, but it is allowed via pay_in_full.
    plan = check_plan_terms(
        NOW,
        balance_jpy=BALANCE,
        months=3,
        down_payment_jpy=BALANCE,
        has_broken_promise=False,
        rules=rules,
    )
    assert plan.allowed is False
    assert "pay_in_full" in plan.reason
    assert _pay(rules)[0] is True


@pytest.mark.parametrize(
    ("amount", "code"),
    [
        (BALANCE + 1, "amount_exceeds_balance"),
        (BALANCE * 2, "amount_exceeds_balance"),
        (BALANCE - 1, "amount_below_balance"),
        (0, "amount_below_balance"),
    ],
)
def test_pay_in_full_amount_must_equal_balance(rules: PolicyRules, amount: int, code: str) -> None:
    assert _pay(rules, amount=amount) == (False, (code,))


@pytest.mark.parametrize(
    ("on", "code"),
    [
        (date(2026, 10, 23), "payment_date_in_past"),
        (date(2026, 11, 8), "payment_date_too_late"),
        (date(2027, 1, 1), "payment_date_too_late"),
    ],
)
def test_pay_in_full_date_window(rules: PolicyRules, on: date, code: str) -> None:
    assert _pay(rules, on=on) == (False, (code,))


def test_pay_in_full_today_uses_debtor_timezone(rules: PolicyRules) -> None:
    # 2026-10-24 02:00 UTC is Oct 24 in Tokyo but Oct 23 in New York.
    now = datetime(2026, 10, 24, 2, tzinfo=UTC)
    assert _pay(rules, now=now, on=date(2026, 10, 23), tz="Asia/Tokyo")[0] is False
    assert _pay(rules, now=now, on=date(2026, 10, 23), tz="America/New_York")[0] is True


def test_pay_in_full_zero_balance_blocked(rules: PolicyRules) -> None:
    ok, codes = _pay(rules, balance=0, amount=0)
    assert ok is False
    assert codes == ("no_balance",)


def test_pay_in_full_invalid_timezone_fails_closed(rules: PolicyRules) -> None:
    assert _pay(rules, tz="Nowhere/Land") == (False, ("invalid_timezone",))


def test_pay_in_full_reports_all_violations(rules: PolicyRules) -> None:
    ok, codes = _pay(rules, amount=BALANCE + 1, on=date(2027, 1, 1))
    assert ok is False
    assert codes == ("amount_exceeds_balance", "payment_date_too_late")
