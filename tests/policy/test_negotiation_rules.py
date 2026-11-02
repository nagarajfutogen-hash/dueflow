import pytest

from app.policy.config import PolicyRules
from app.policy.engine import (
    Rule,
    check_history_access,
    check_no_discount,
    check_open_dispute,
    check_plan_terms,
)
from tests.policy.conftest import jst

NOW = jst(2026, 10, 24, 12)
BALANCE = 100_000


def _plan(
    rules: PolicyRules,
    months: int = 3,
    down: int = 10_000,
    broken: bool = False,
    balance: int = BALANCE,
) -> tuple[bool, tuple[str, ...]]:
    d = check_plan_terms(
        NOW,
        balance_jpy=balance,
        months=months,
        down_payment_jpy=down,
        has_broken_promise=broken,
        rules=rules,
    )
    assert d.rule is Rule.PLAN_TERMS
    return d.allowed, d.codes


# --- Plan length ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("months", "allowed"),
    [(-1, False), (0, False), (1, False), (2, False), (3, True), (4, True), (5, True), (6, True),
     (7, False), (12, False)],
)  # fmt: skip
def test_plan_months_bounds(rules: PolicyRules, months: int, allowed: bool) -> None:
    ok, codes = _plan(rules, months=months)
    assert ok is allowed
    assert codes == (() if allowed else ("months_out_of_range",))


# --- Down payment -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("down", "broken", "allowed"),
    [
        (9_999, False, False),
        (10_000, False, True),
        (50_000, False, True),
        (0, False, False),
        (-1, False, False),
        (19_999, True, False),
        (20_000, True, True),
        (10_000, True, False),
    ],
)
def test_down_payment_minimum(rules: PolicyRules, down: int, broken: bool, allowed: bool) -> None:
    ok, codes = _plan(rules, down=down, broken=broken)
    assert ok is allowed
    assert codes == (() if allowed else ("down_payment_too_low",))


def test_down_payment_rounds_up_to_whole_yen(rules: PolicyRules) -> None:
    # 10% of 99,999 = 9,999.9 -> at least 10,000 yen.
    assert _plan(rules, balance=99_999, down=9_999)[0] is False
    assert _plan(rules, balance=99_999, down=10_000)[0] is True


def test_down_payment_equal_to_balance_blocked(rules: PolicyRules) -> None:
    ok, codes = _plan(rules, down=BALANCE)
    assert ok is False
    assert codes == ("down_payment_covers_balance",)


def test_down_payment_above_balance_blocked(rules: PolicyRules) -> None:
    ok, codes = _plan(rules, down=BALANCE + 1)
    assert ok is False
    assert "down_payment_covers_balance" in codes


def test_multiple_violations_all_reported(rules: PolicyRules) -> None:
    ok, codes = _plan(rules, months=7, down=0)
    assert ok is False
    assert codes == ("months_out_of_range", "down_payment_too_low")


def test_zero_balance_blocked(rules: PolicyRules) -> None:
    ok, codes = _plan(rules, balance=0, down=0)
    assert ok is False
    assert "no_balance" in codes


def test_plan_decision_records_required_minimum(rules: PolicyRules) -> None:
    d = check_plan_terms(
        NOW,
        balance_jpy=BALANCE,
        months=3,
        down_payment_jpy=5_000,
        has_broken_promise=True,
        rules=rules,
    )
    assert d.input["min_down_payment_jpy"] == 20_000
    assert d.input["rules_version"] == rules.version
    assert "20%" in d.reason


# --- No discount ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("offered", "allowed", "code"),
    [
        (BALANCE, True, None),
        (BALANCE - 1, False, "discount_not_allowed"),
        (BALANCE // 2, False, "discount_not_allowed"),
        (0, False, "discount_not_allowed"),
        (BALANCE + 1, False, "total_exceeds_balance"),
    ],
)
def test_no_discount(offered: int, allowed: bool, code: str | None) -> None:
    d = check_no_discount(NOW, balance_jpy=BALANCE, offered_total_jpy=offered)
    assert d.rule is Rule.NO_DISCOUNT
    assert d.allowed is allowed
    assert d.codes == ((code,) if code else ())


# --- Open dispute -----------------------------------------------------------------------


def test_open_dispute_blocks_negotiation() -> None:
    d = check_open_dispute(NOW, has_open_dispute=True)
    assert d.allowed is False
    assert d.codes == ("open_dispute",)


def test_no_dispute_allows_negotiation() -> None:
    assert check_open_dispute(NOW, has_open_dispute=False).allowed is True


# --- History access ---------------------------------------------------------------------


def test_history_blocked_without_verification() -> None:
    d = check_history_access(NOW, identity_verified=False)
    assert d.allowed is False
    assert d.codes == ("identity_not_verified",)


def test_history_allowed_after_verification() -> None:
    assert check_history_access(NOW, identity_verified=True).allowed is True
