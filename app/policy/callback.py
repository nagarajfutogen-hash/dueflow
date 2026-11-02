"""Agent v5 (rules-v8): customer-requested callbacks. Pure: no LLM, no database, no clock.

A callback date must come from the customer's own words in this call, never from the LLM
alone (the D-001 lesson: an LLM fills a date argument with something nobody said). Code works
out every date the customer named and allows the requested date only if it is one of them,
between debtor-local tomorrow and `callback.max_days_ahead` days ahead.
"""

import calendar
import re
import unicodedata
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.policy.config import PolicyRules
from app.policy.engine import Decision, Rule
from app.policy.identity import find_dates
from app.policy.output_check import MONTHS

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_WEEKDAY = re.compile(rf"\b(next\s+|this\s+|coming\s+)?({'|'.join(_WEEKDAYS)})\b", re.I)
_NEXT_WEEK = re.compile(r"\bnext\s+week\b", re.I)
_NEXT_MONTH = re.compile(r"\bnext\s+month\b", re.I)
_END_OF_MONTH = re.compile(r"\bend\s+of\s+(?:the\s+)?month\b", re.I)
_TOMORROW = re.compile(r"\btomorrow\b", re.I)
_TODAY = re.compile(r"\btoday\b", re.I)
_MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))
_ORD = r"(?:st|nd|rd|th)?"
# A year given ("January 20th, 2027"): find_dates reads it; blanked so "20th" is not re-read.
_WITH_YEAR = re.compile(
    rf"\b(?:(?:{_MONTH_ALT})\.?\s+\d{{1,2}}{_ORD},?\s+\d{{4}}"
    rf"|\d{{1,2}}{_ORD}\s+(?:of\s+)?(?:{_MONTH_ALT}),?\s+\d{{4}})\b",
    re.I,
)
# V5-5: a month and day without a year ("January 20th" was read as the next 20th).
_MONTH_DAY = re.compile(rf"\b({_MONTH_ALT})\.?\s+(?:the\s+)?(\d{{1,2}}){_ORD}\b", re.I)
_DAY_MONTH = re.compile(rf"\b(?:the\s+)?(\d{{1,2}}){_ORD}\s+of\s+({_MONTH_ALT})\b", re.I)
_DAY_OF_MONTH = re.compile(r"\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\b", re.I)
_IN_DAYS = re.compile(r"\bin\s+(\d{1,2}|a|one|two|three|four)\s+(day|days|week|weeks)\b", re.I)
_SMALL = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4}


def _month_days(year: int, month: int) -> list[date]:
    return [date(year, month, d) for d in range(1, calendar.monthrange(year, month)[1] + 1)]


def _next_month(today: date) -> tuple[int, int]:
    return (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)


def customer_dates(said: Sequence[str], today: date) -> dict[date, str]:
    """Every date the customer named, with the phrase it came from. `today` is debtor-local.
    Relative words are read forward from today: 'Friday' is the next Friday, 'the 10th' the
    next 10th, 'next week' any day of next calendar week. 'next Friday' allows both the next
    Friday and the one after (people use it both ways). 'January 20th' is the next January 20,
    never the next 20th of any month. 'today' is today (a callback still refuses it)."""
    found: dict[date, str] = {}

    def add(d: date, phrase: str) -> None:
        found.setdefault(d, phrase.lower())

    for raw in said:
        text = " ".join(unicodedata.normalize("NFKC", raw).split())
        for d in find_dates(text):
            add(d, d.isoformat())
        text = _WITH_YEAR.sub(" ", text)
        for pattern, month_group, day_group in ((_MONTH_DAY, 1, 2), (_DAY_MONTH, 2, 1)):
            for m in pattern.finditer(text):
                month, day = MONTHS[m[month_group].lower()], int(m[day_group])
                for year in (today.year, today.year + 1):
                    real = day <= calendar.monthrange(year, month)[1]
                    if real and date(year, month, day) > today:
                        add(date(year, month, day), m.group(0))
                        break
            text = pattern.sub(" ", text)
        for m in _WEEKDAY.finditer(text):
            ahead = (_WEEKDAYS.index(m[2].lower()) - today.weekday()) % 7 or 7
            add(today + timedelta(days=ahead), m.group(0))
            if m[1] and m[1].strip().lower() == "next":
                add(today + timedelta(days=ahead + 7), m.group(0))
        if week := _NEXT_WEEK.search(text):
            monday = today + timedelta(days=7 - today.weekday())
            for i in range(7):
                add(monday + timedelta(days=i), week.group(0))
        if next_month := _NEXT_MONTH.search(text):
            for d in _month_days(*_next_month(today)):
                add(d, next_month.group(0))
        if end := _END_OF_MONTH.search(text):
            last = _month_days(today.year, today.month)[-1]
            add(last if last > today else _month_days(*_next_month(today))[-1], end.group(0))
        if today_word := _TODAY.search(text):
            add(today, today_word.group(0))
        if tomorrow := _TOMORROW.search(text):
            add(today + timedelta(days=1), tomorrow.group(0))
        for m in _DAY_OF_MONTH.finditer(text):
            day = int(m[1])
            for year, month in ((today.year, today.month), _next_month(today)):
                if day <= calendar.monthrange(year, month)[1] and date(year, month, day) > today:
                    add(date(year, month, day), m.group(0))
                    break
        for m in _IN_DAYS.finditer(text):
            count = int(m[1]) if m[1].isdigit() else _SMALL[m[1].lower()]
            days = count * (7 if m[2].lower().startswith("week") else 1)
            add(today + timedelta(days=days), m.group(0))
    return found


def named_payment_date(
    day_phrase: str, said: Sequence[str], today: date
) -> tuple[date, str] | None:
    """V5-5 pay in full: the earliest date from today on that `day_phrase` names AND the
    customer said in this call, with their words; None if there is none. The LLM passes words,
    never a date (call 198: the LLM sent a past date for 'Friday')."""
    named = customer_dates(said, today)
    for d in sorted(customer_dates([day_phrase], today)):
        if d >= today and d in named:
            return d, named[d]
    return None


def resolve_pay_in_full_date(
    now: datetime,
    *,
    debtor_tz: str,
    day_phrase: str,
    said: Sequence[str],
    rules: PolicyRules,
) -> tuple[date | None, Decision | None]:
    """V5-5: the payment date for a pay-in-full request, from the customer's words.
    Resolved: (date, None); check_pay_in_full then decides on that date (one terms event).
    Not resolved: (None, a blocked pay_in_full decision with code date_not_said)."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    today = now.astimezone(ZoneInfo(debtor_tz)).date()
    found = named_payment_date(day_phrase, said, today)
    if found is not None:
        return found[0], None
    return None, Decision(
        rule=Rule.PAY_IN_FULL,
        allowed=False,
        reason="the customer did not name this day in this call",
        at=now,
        input={
            "day_phrase": day_phrase[:60],
            "debtor_local_date": today.isoformat(),
            "rules_version": rules.version,
        },
        codes=("date_not_said",),
    )


def check_callback(
    now: datetime,
    *,
    debtor_tz: str,
    requested: date,
    said: Sequence[str],
    rules: PolicyRules,
) -> Decision:
    """Allowed when the requested date is one the customer named in this call and falls
    between debtor-local tomorrow and today + callback.max_days_ahead."""
    if rules.callback is None:
        raise ValueError(f"{rules.version} has no callback block (rules-v8 or later)")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    today = now.astimezone(ZoneInfo(debtor_tz)).date()
    latest = today + timedelta(days=rules.callback.max_days_ahead)
    named = customer_dates(said, today)
    inp: dict[str, Any] = {
        "requested": requested.isoformat(),
        "debtor_local_today": today.isoformat(),
        "max_days_ahead": rules.callback.max_days_ahead,
        "customer_phrase": named.get(requested),
        "rules_version": rules.version,
    }
    codes: list[str] = []
    reasons: list[str] = []
    if requested <= today:
        codes.append("too_soon")
        reasons.append("a callback must be from tomorrow (one call a day)")
    if requested > latest:
        codes.append("too_far")
        reasons.append(f"a callback must be within {rules.callback.max_days_ahead} days")
    if requested not in named:
        codes.append("date_not_said")
        reasons.append("the customer did not name this date in this call")
    if codes:
        return Decision(
            rule=Rule.CALLBACK_REQUEST,
            allowed=False,
            reason="; ".join(reasons),
            at=now,
            input=inp,
            codes=tuple(codes),
        )
    return Decision(
        rule=Rule.CALLBACK_REQUEST,
        allowed=True,
        reason=f"callback on a date the customer named ({named[requested]})",
        at=now,
        input=inp,
    )
