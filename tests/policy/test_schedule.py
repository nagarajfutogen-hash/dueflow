from datetime import date

import pytest

from app.policy.engine import Rule, check_identity_required
from app.policy.schedule import add_months, full_payment_schedule, plan_schedule
from tests.policy.conftest import jst


@pytest.mark.parametrize(
    ("start", "months", "expected"),
    [
        (date(2026, 10, 31), 1, date(2026, 11, 30)),
        (date(2027, 1, 31), 1, date(2027, 2, 28)),
        (date(2028, 1, 31), 1, date(2028, 2, 29)),  # leap year
        (date(2026, 12, 15), 1, date(2027, 1, 15)),
        (date(2026, 10, 31), 12, date(2027, 10, 31)),
    ],
)
def test_add_months_clamps_to_month_end(start: date, months: int, expected: date) -> None:
    assert add_months(start, months) == expected


@pytest.mark.parametrize("balance", [100_000, 99_999, 123_457, 300_001])
@pytest.mark.parametrize("months", [3, 4, 5, 6])
@pytest.mark.parametrize("down_pct", [10, 20, 55])
def test_plan_schedule_always_totals_balance(balance: int, months: int, down_pct: int) -> None:
    down = -(-balance * down_pct // 100)
    schedule = plan_schedule(
        balance_jpy=balance,
        months=months,
        down_payment_jpy=down,
        down_payment_due=date(2026, 10, 31),
    )
    assert sum(p.amount_jpy for p in schedule) == balance
    assert len(schedule) == months + 1
    assert schedule[0].kind == "down_payment"
    installments = [p.amount_jpy for p in schedule[1:]]
    assert len(set(installments[:-1])) <= 1  # all equal except possibly the last
    assert all(a > 0 for a in installments)


def test_plan_schedule_dates_are_monthly_after_down_payment() -> None:
    schedule = plan_schedule(
        balance_jpy=100_000, months=3, down_payment_jpy=10_000, down_payment_due=date(2026, 10, 31)
    )
    assert [p.due for p in schedule] == [
        date(2026, 10, 31),
        date(2026, 11, 30),
        date(2026, 12, 31),
        date(2027, 1, 31),
    ]
    assert [p.amount_jpy for p in schedule] == [10_000, 30_000, 30_000, 30_000]


def test_last_installment_absorbs_remainder() -> None:
    schedule = plan_schedule(
        balance_jpy=100_001, months=3, down_payment_jpy=10_001, down_payment_due=date(2026, 11, 1)
    )
    assert [p.amount_jpy for p in schedule[1:]] == [30_000, 30_000, 30_000]
    schedule = plan_schedule(
        balance_jpy=100_002, months=3, down_payment_jpy=10_000, down_payment_due=date(2026, 11, 1)
    )
    assert [p.amount_jpy for p in schedule[1:]] == [30_000, 30_000, 30_002]


def test_full_payment_schedule() -> None:
    [p] = full_payment_schedule(amount_jpy=50_000, due=date(2026, 11, 1))
    assert (p.amount_jpy, p.kind) == (50_000, "full_payment")


def test_identity_required_blocks_unverified() -> None:
    d = check_identity_required(jst(2026, 10, 24, 12), identity_verified=False, action="x")
    assert d.rule is Rule.IDENTITY_REQUIRED
    assert (d.allowed, d.codes) == (False, ("identity_not_verified",))


def test_identity_required_allows_verified() -> None:
    d = check_identity_required(jst(2026, 10, 24, 12), identity_verified=True, action="x")
    assert d.allowed is True
