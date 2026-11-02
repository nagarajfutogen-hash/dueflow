"""Pure payment-schedule math. Integer yen only; totals always equal the balance."""

import calendar
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Payment:
    due: date
    amount_jpy: int
    kind: str  # "down_payment" | "installment" | "full_payment"


def add_months(d: date, months: int) -> date:
    """Same day-of-month `months` later, clamped to the month's last day (Jan 31 -> Feb 28)."""
    index = d.month - 1 + months
    year, month = d.year + index // 12, index % 12 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def plan_schedule(
    *, balance_jpy: int, months: int, down_payment_jpy: int, down_payment_due: date
) -> list[Payment]:
    """Down payment, then `months` monthly installments; the last one absorbs the remainder."""
    remaining = balance_jpy - down_payment_jpy
    base = remaining // months
    schedule = [Payment(down_payment_due, down_payment_jpy, "down_payment")]
    for i in range(1, months + 1):
        amount = base if i < months else remaining - base * (months - 1)
        schedule.append(Payment(add_months(down_payment_due, i), amount, "installment"))
    return schedule


def full_payment_schedule(*, amount_jpy: int, due: date) -> list[Payment]:
    return [Payment(due, amount_jpy, "full_payment")]
