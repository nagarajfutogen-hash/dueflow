"""Deterministic policy engine.

Pure functions only: no LLM, no network, no database, no file I/O, no wall clock.
The current time is always passed in as `now` (timezone-aware). Each function returns
a Decision; recording it to `events` is the caller's job (app/db/audit.py).
"""

import enum
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.policy.claims import find_phrase, name_variants
from app.policy.config import PolicyRules
from app.policy.identity import find_dates, name_in_text


class Rule(enum.StrEnum):
    CONTACT_WINDOW = "contact_window"
    ATTEMPT_LIMITS = "attempt_limits"
    STOP_CONTACT = "stop_contact"
    WORKPLACE_CONTACT = "workplace_contact"
    PLAN_TERMS = "plan_terms"
    PAY_IN_FULL = "pay_in_full"
    NO_DISCOUNT = "no_discount"
    OPEN_DISPUTE = "open_dispute"
    HISTORY_ACCESS = "history_access"
    IDENTITY_VERIFICATION = "identity_verification"
    IDENTITY_REQUIRED = "identity_required"
    OUTPUT_CHECK = "output_check"
    WRONG_NUMBER_CLAIM = "wrong_number_claim"  # rules-v7
    PAYMENT_CLAIM = "payment_claim"  # rules-v7
    TERMS_PRESENTED = "terms_presented"  # agent v4: terms said before a promise
    PAYMENT_RECORD = "payment_record"  # agent v4: a "paid" claim checked against records
    NEGOTIATION_LADDER = "negotiation_ladder"  # agent v5: the engine picks the next offer
    CALLBACK_REQUEST = "callback_request"  # agent v5: a callback on a date the customer named


class DialTarget(enum.StrEnum):
    HOME = "home"
    MOBILE = "mobile"
    WORK = "work"


@dataclass(frozen=True)
class Decision:
    rule: Rule
    allowed: bool
    reason: str
    at: datetime
    input: dict[str, Any]
    # Machine-readable reasons; empty when allowed.
    codes: tuple[str, ...] = field(default=())


def _require_aware(now: datetime) -> None:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")


def _allow(rule: Rule, now: datetime, reason: str, inp: dict[str, Any]) -> Decision:
    return Decision(rule=rule, allowed=True, reason=reason, at=now, input=inp)


def _block(
    rule: Rule, now: datetime, codes: Sequence[str], reasons: Sequence[str], inp: dict[str, Any]
) -> Decision:
    return Decision(
        rule=rule, allowed=False, reason="; ".join(reasons), at=now, input=inp, codes=tuple(codes)
    )


def _zone(tz_name: str) -> ZoneInfo | None:
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


# --- Contact rules -------------------------------------------------------------------


def check_contact_window(now: datetime, debtor_tz: str, rules: PolicyRules) -> Decision:
    """Art. 21: contact only between start (inclusive) and end (exclusive), debtor local time."""
    _require_aware(now)
    window = rules.contact_window
    inp: dict[str, Any] = {
        "now": now.isoformat(),
        "debtor_tz": debtor_tz,
        "window": f"{window.start:%H:%M}-{window.end:%H:%M}",
        "rules_version": rules.version,
    }
    zone = _zone(debtor_tz)
    if zone is None:
        # Fail closed: without a valid timezone we cannot know the debtor's local time.
        return _block(
            Rule.CONTACT_WINDOW,
            now,
            ["invalid_timezone"],
            [f"unknown timezone {debtor_tz!r}"],
            inp,
        )
    local = now.astimezone(zone)
    inp["debtor_local_time"] = local.isoformat()
    if window.start <= local.time() < window.end:
        return _allow(Rule.CONTACT_WINDOW, now, "inside contact window", inp)
    return _block(
        Rule.CONTACT_WINDOW,
        now,
        ["outside_contact_window"],
        [f"debtor local time {local:%H:%M} is outside {inp['window']}"],
        inp,
    )


def check_attempt_limits(
    now: datetime, debtor_tz: str, prior_attempts: Sequence[datetime], rules: PolicyRules
) -> Decision:
    """Limits per debtor-local calendar day and per rolling 7 days (now - 7d, now]."""
    _require_aware(now)
    for attempt in prior_attempts:
        _require_aware(attempt)
    limits = rules.attempt_limits
    inp: dict[str, Any] = {
        "now": now.isoformat(),
        "debtor_tz": debtor_tz,
        "prior_attempts": [a.isoformat() for a in prior_attempts],
        "rules_version": rules.version,
    }
    zone = _zone(debtor_tz)
    if zone is None:
        return _block(
            Rule.ATTEMPT_LIMITS,
            now,
            ["invalid_timezone"],
            [f"unknown timezone {debtor_tz!r}"],
            inp,
        )
    today = now.astimezone(zone).date()
    week_start = now - timedelta(days=7)
    past = [a for a in prior_attempts if a <= now]
    same_day = sum(1 for a in past if a.astimezone(zone).date() == today)
    last_7_days = sum(1 for a in past if a > week_start)
    inp |= {"attempts_today": same_day, "attempts_last_7_days": last_7_days}

    codes: list[str] = []
    reasons: list[str] = []
    if same_day >= limits.per_local_day:
        codes.append("daily_limit_reached")
        reasons.append(f"{same_day} attempt(s) today; limit {limits.per_local_day}/day")
    if last_7_days >= limits.per_rolling_7_days:
        codes.append("weekly_limit_reached")
        reasons.append(
            f"{last_7_days} attempt(s) in 7 days; limit {limits.per_rolling_7_days}/7 days"
        )
    if codes:
        return _block(Rule.ATTEMPT_LIMITS, now, codes, reasons, inp)
    return _allow(Rule.ATTEMPT_LIMITS, now, "within attempt limits", inp)


def check_stop_contact(now: datetime, stop_contact: bool) -> Decision:
    """Stop-contact is permanent once requested."""
    _require_aware(now)
    inp: dict[str, Any] = {"stop_contact": stop_contact}
    if stop_contact:
        return _block(
            Rule.STOP_CONTACT,
            now,
            ["stop_contact_requested"],
            ["debtor asked to stop contact"],
            inp,
        )
    return _allow(Rule.STOP_CONTACT, now, "no stop-contact request", inp)


def check_workplace_contact(
    now: datetime,
    target: DialTarget,
    no_workplace_contact: bool,
    justification: str | None,
) -> Decision:
    """Art. 21: no workplace contact without a recorded justification.

    A debtor's "don't call my work" is a permanent flag that overrides any justification.
    """
    _require_aware(now)
    has_justification = bool(justification and justification.strip())
    inp: dict[str, Any] = {
        "target": target.value,
        "no_workplace_contact": no_workplace_contact,
        "has_justification": has_justification,
    }
    if target is not DialTarget.WORK:
        return _allow(Rule.WORKPLACE_CONTACT, now, "not a workplace number", inp)
    if no_workplace_contact:
        return _block(
            Rule.WORKPLACE_CONTACT,
            now,
            ["debtor_refused_workplace_contact"],
            ["debtor asked not to be contacted at work"],
            inp,
        )
    if not has_justification:
        return _block(
            Rule.WORKPLACE_CONTACT,
            now,
            ["no_workplace_justification"],
            ["workplace contact requires a recorded justification"],
            inp,
        )
    return _allow(Rule.WORKPLACE_CONTACT, now, "workplace contact justified", inp)


def pre_call_check(
    now: datetime,
    *,
    debtor_tz: str,
    prior_attempts: Sequence[datetime],
    stop_contact: bool,
    target: DialTarget,
    no_workplace_contact: bool,
    workplace_justification: str | None,
    rules: PolicyRules,
) -> list[Decision]:
    """Evaluate every contact rule (all are recorded). The dial is allowed only if all allow."""
    return [
        check_stop_contact(now, stop_contact),
        check_contact_window(now, debtor_tz, rules),
        check_attempt_limits(now, debtor_tz, prior_attempts, rules),
        check_workplace_contact(now, target, no_workplace_contact, workplace_justification),
    ]


# --- Negotiation rules ---------------------------------------------------------------


def _ceil_pct(amount: int, pct: int) -> int:
    """Smallest integer yen >= amount * pct / 100 (integer math, no floats)."""
    return -(-amount * pct // 100)


def min_down_payment(balance_jpy: int, *, has_broken_promise: bool, rules: PolicyRules) -> int:
    """The smallest down payment any plan allows (higher after a broken promise)."""
    plan = rules.plan
    pct = (
        plan.min_down_payment_pct_after_broken_promise
        if has_broken_promise
        else plan.min_down_payment_pct
    )
    return _ceil_pct(balance_jpy, pct)


@dataclass(frozen=True)
class CustomerLimits:
    """Agent v5 (rules-v8): one customer's own plan limits from the file. Each may only be
    stricter than the global plan rules; None = the global rule applies."""

    min_monthly_jpy: int | None = None
    max_months: int | None = None

    def as_input(self) -> dict[str, int | None]:
        return {"min_monthly_jpy": self.min_monthly_jpy, "max_months": self.max_months}


def check_plan_terms(
    now: datetime,
    *,
    balance_jpy: int,
    months: int,
    down_payment_jpy: int,
    has_broken_promise: bool,
    rules: PolicyRules,
    customer: CustomerLimits | None = None,
) -> Decision:
    """Approved plans: min..max installments, down payment >= pct of balance, balance unchanged.

    customer (agent v5 only): the customer's own stricter limits, checked on top. None keeps
    the v1-v4 decision and its recorded input exactly as before."""
    _require_aware(now)
    plan = rules.plan
    pct = (
        plan.min_down_payment_pct_after_broken_promise
        if has_broken_promise
        else plan.min_down_payment_pct
    )
    min_down = _ceil_pct(balance_jpy, pct)
    inp: dict[str, Any] = {
        "balance_jpy": balance_jpy,
        "months": months,
        "down_payment_jpy": down_payment_jpy,
        "has_broken_promise": has_broken_promise,
        "min_down_payment_jpy": min_down,
        "rules_version": rules.version,
    }
    codes: list[str] = []
    reasons: list[str] = []
    if balance_jpy <= 0:
        codes.append("no_balance")
        reasons.append("nothing is owed")
    if not plan.min_months <= months <= plan.max_months:
        codes.append("months_out_of_range")
        reasons.append(f"{months} installments; allowed {plan.min_months}-{plan.max_months}")
    if down_payment_jpy < min_down:
        codes.append("down_payment_too_low")
        reasons.append(f"down payment ¥{down_payment_jpy:,} < required ¥{min_down:,} ({pct}%)")
    if balance_jpy > 0 and down_payment_jpy >= balance_jpy:
        codes.append("down_payment_covers_balance")
        reasons.append("down payment covers the full balance; use pay_in_full instead")
    if customer is not None:
        inp["customer_limits"] = customer.as_input()
        if customer.max_months is not None and months > customer.max_months:
            codes.append("above_customer_max_months")
            reasons.append(
                f"{months} installments; this customer allows at most {customer.max_months}"
            )
        monthly = (balance_jpy - down_payment_jpy) // months if months > 0 else 0
        if customer.min_monthly_jpy is not None and monthly < customer.min_monthly_jpy:
            codes.append("below_customer_min_monthly")
            reasons.append(
                f"monthly ¥{monthly:,} < this customer's minimum ¥{customer.min_monthly_jpy:,}"
            )
    if codes:
        return _block(Rule.PLAN_TERMS, now, codes, reasons, inp)
    return _allow(Rule.PLAN_TERMS, now, "plan within approved terms", inp)


def check_pay_in_full(
    now: datetime,
    *,
    debtor_tz: str,
    balance_jpy: int,
    amount_jpy: int,
    payment_date: date,
    rules: PolicyRules,
) -> Decision:
    """Single payment of exactly the balance, dated debtor-local today .. today + N days."""
    _require_aware(now)
    max_days = rules.pay_in_full.max_days_until_payment
    inp: dict[str, Any] = {
        "now": now.isoformat(),
        "debtor_tz": debtor_tz,
        "balance_jpy": balance_jpy,
        "amount_jpy": amount_jpy,
        "payment_date": payment_date.isoformat(),
        "rules_version": rules.version,
    }
    zone = _zone(debtor_tz)
    if zone is None:
        return _block(
            Rule.PAY_IN_FULL, now, ["invalid_timezone"], [f"unknown timezone {debtor_tz!r}"], inp
        )
    today = now.astimezone(zone).date()
    latest = today + timedelta(days=max_days)
    inp |= {"debtor_local_date": today.isoformat(), "latest_payment_date": latest.isoformat()}

    codes: list[str] = []
    reasons: list[str] = []
    if balance_jpy <= 0:
        codes.append("no_balance")
        reasons.append("nothing is owed")
    if amount_jpy < balance_jpy:
        codes.append("amount_below_balance")
        reasons.append(f"¥{amount_jpy:,} is less than balance ¥{balance_jpy:,}")
    elif amount_jpy > balance_jpy:
        codes.append("amount_exceeds_balance")
        reasons.append(f"¥{amount_jpy:,} exceeds balance ¥{balance_jpy:,}")
    if payment_date < today:
        codes.append("payment_date_in_past")
        reasons.append(f"payment date {payment_date} is before today ({today})")
    elif payment_date > latest:
        codes.append("payment_date_too_late")
        reasons.append(f"payment date {payment_date} is after {latest} ({max_days} days)")
    if codes:
        return _block(Rule.PAY_IN_FULL, now, codes, reasons, inp)
    return _allow(Rule.PAY_IN_FULL, now, "full payment within policy", inp)


def check_no_discount(
    now: datetime, *, balance_jpy: int, offered_total_jpy: int, reject_zero_offer: bool = False
) -> Decision:
    """The total to be paid must equal the balance: no principal waiver or discount.

    reject_zero_offer (agent v3, fixes.payment_claim_handoff): an offer of ¥0 or less is not a
    discount request (in v2 it was an LLM's misreading of "I already paid")."""
    _require_aware(now)
    inp: dict[str, Any] = {"balance_jpy": balance_jpy, "offered_total_jpy": offered_total_jpy}
    if reject_zero_offer and offered_total_jpy <= 0:
        return _block(
            Rule.NO_DISCOUNT,
            now,
            ["invalid_offer"],
            [f"an offer of ¥{offered_total_jpy:,} is not a discount request"],
            inp | {"reject_zero_offer": True},
        )
    if offered_total_jpy < balance_jpy:
        return _block(
            Rule.NO_DISCOUNT,
            now,
            ["discount_not_allowed"],
            [f"offered ¥{offered_total_jpy:,} is less than balance ¥{balance_jpy:,}"],
            inp,
        )
    if offered_total_jpy > balance_jpy:
        return _block(
            Rule.NO_DISCOUNT,
            now,
            ["total_exceeds_balance"],
            [f"offered ¥{offered_total_jpy:,} exceeds balance ¥{balance_jpy:,}"],
            inp,
        )
    return _allow(Rule.NO_DISCOUNT, now, "total equals balance", inp)


def check_open_dispute(now: datetime, has_open_dispute: bool) -> Decision:
    """An open dispute blocks negotiation; the conversation goes to handoff."""
    _require_aware(now)
    inp: dict[str, Any] = {"has_open_dispute": has_open_dispute}
    if has_open_dispute:
        return _block(
            Rule.OPEN_DISPUTE,
            now,
            ["open_dispute"],
            ["debt is disputed; negotiation blocked, hand off"],
            inp,
        )
    return _allow(Rule.OPEN_DISPUTE, now, "no open dispute", inp)


# --- Memory --------------------------------------------------------------------------


def check_history_access(now: datetime, identity_verified: bool) -> Decision:
    """Account history only after identity is verified in the current conversation."""
    _require_aware(now)
    inp: dict[str, Any] = {"identity_verified": identity_verified}
    if not identity_verified:
        return _block(
            Rule.HISTORY_ACCESS,
            now,
            ["identity_not_verified"],
            ["identity not verified in this conversation"],
            inp,
        )
    return _allow(Rule.HISTORY_ACCESS, now, "identity verified", inp)


# --- Claims in the person's own words (rules-v7) ------------------------------------------


def check_wrong_number_claim(
    now: datetime, *, debtor_text: str, debtor_name: str, rules: PolicyRules
) -> Decision:
    """A call may end as a wrong number only if the person said so in their own words.

    `debtor_text` is everything the person said in this call. The input records the matched
    phrase as written in the rules ({name} unfilled), never what the person said."""
    _require_aware(now)
    if rules.third_party is None:
        raise ValueError(f"{rules.version} has no third_party phrase list")
    phrase = find_phrase(
        debtor_text, rules.third_party.wrong_number_phrases, names=name_variants(debtor_name)
    )
    inp: dict[str, Any] = {"matched_phrase": phrase, "rules_version": rules.version}
    if phrase is None:
        return _block(
            Rule.WRONG_NUMBER_CLAIM,
            now,
            ["wrong_number_not_stated"],
            ["the person has not said this is a wrong number"],
            inp,
        )
    return _allow(Rule.WRONG_NUMBER_CLAIM, now, "the person said this is a wrong number", inp)


def check_payment_claim(now: datetime, *, debtor_text: str, rules: PolicyRules) -> Decision:
    """After verification: a claim that the debt is already paid blocks negotiation and goes to
    a dispute handoff. `debtor_text` is the person's latest turn."""
    _require_aware(now)
    if rules.dispute is None:
        raise ValueError(f"{rules.version} has no dispute phrase list")
    phrase = find_phrase(debtor_text, rules.dispute.payment_claim_phrases)
    inp: dict[str, Any] = {"matched_phrase": phrase, "rules_version": rules.version}
    if phrase is not None:
        return _block(
            Rule.PAYMENT_CLAIM,
            now,
            ["payment_claimed"],
            ["the person says the debt is already paid: dispute handoff required"],
            inp,
        )
    return _allow(Rule.PAYMENT_CLAIM, now, "no payment claim", inp)


@dataclass(frozen=True)
class PaymentOnFile:
    amount_jpy: int
    paid_on: date  # debtor-local date the payment was received
    method: str | None = None
    reference: str | None = None


@dataclass(frozen=True)
class PaymentClaim:
    paid_on: date | None = None
    amount_jpy: int | None = None
    method: str | None = None
    reference: str | None = None


def _ref(value: str | None) -> str:
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())


def check_payment_record(
    now: datetime, *, claim: PaymentClaim, payments: Sequence[PaymentOnFile]
) -> Decision:
    """Agent v4: compare a "I paid" claim with payments on record. Allowed when one payment
    matches by transaction reference, by amount and date, or by amount alone (the person may
    misremember the date: input match='amount_only', the agent asks which one). Blocked when
    nothing matches: never proof the person is wrong (it may still be processing)."""
    _require_aware(now)
    inp: dict[str, Any] = {
        "claim": {
            "paid_on": claim.paid_on.isoformat() if claim.paid_on else None,
            "amount_jpy": claim.amount_jpy,
            "method": claim.method,
            "reference": claim.reference,
        },
        "payments_on_file": len(payments),
    }
    if claim.paid_on is None and claim.amount_jpy is None and not _ref(claim.reference):
        return _block(
            Rule.PAYMENT_RECORD,
            now,
            ["claim_details_missing"],
            ["no date, amount or transaction reference given"],
            inp,
        )
    found: tuple[str, PaymentOnFile] | None = None
    for p in payments:
        if _ref(claim.reference) and _ref(claim.reference) == _ref(p.reference):
            found = ("reference", p)
            break
        if claim.amount_jpy == p.amount_jpy and claim.paid_on == p.paid_on:
            found = ("amount_and_date", p)
            break
    if found is None:
        same_amount = [p for p in payments if claim.amount_jpy == p.amount_jpy]
        if same_amount:
            found = ("amount_only", same_amount[-1])
    if found is None:
        return _block(
            Rule.PAYMENT_RECORD,
            now,
            ["no_matching_payment"],
            ["no payment on record matches the claim"],
            inp,
        )
    match, p = found
    inp |= {
        "match": match,
        "payment": {
            "amount_jpy": p.amount_jpy,
            "paid_on": p.paid_on.isoformat(),
            "method": p.method,
        },
    }
    return _allow(Rule.PAYMENT_RECORD, now, f"payment on record ({match})", inp)


# --- Identity --------------------------------------------------------------------------


def check_identity_required(now: datetime, *, identity_verified: bool, action: str) -> Decision:
    """Anything that discusses the debt (terms, promises) needs a verified identity first."""
    _require_aware(now)
    inp: dict[str, Any] = {"identity_verified": identity_verified, "action": action}
    if not identity_verified:
        return _block(
            Rule.IDENTITY_REQUIRED,
            now,
            ["identity_not_verified"],
            [f"{action} requires a verified identity"],
            inp,
        )
    return _allow(Rule.IDENTITY_REQUIRED, now, "identity verified", inp)


def check_identity(
    now: datetime,
    *,
    debtor_text: str,
    actual_name: str,
    actual_dob: date,
    failed_attempts: int,
    rules: PolicyRules,
) -> Decision:
    """Verify from the debtor's OWN words, never from values an LLM passes along.

    `debtor_text` is what the person said since the last attempt. No date in it at all is not
    an attempt (code `no_details`); any date that is wrong counts as a failed attempt.
    The decision input records only match booleans, never names or dates.
    """
    _require_aware(now)
    max_failed = rules.identity.max_failed_attempts
    inp: dict[str, Any] = {
        "attempt": failed_attempts + 1,
        "max_failed_attempts": max_failed,
        "rules_version": rules.version,
    }
    if failed_attempts >= max_failed:
        return _block(
            Rule.IDENTITY_VERIFICATION,
            now,
            ["max_attempts_reached"],
            ["no verification attempts left"],
            inp,
        )
    dates = find_dates(debtor_text)
    if not dates:
        inp["attempt"] = None
        return _block(
            Rule.IDENTITY_VERIFICATION,
            now,
            ["no_details"],
            ["no date of birth stated yet; not counted as an attempt"],
            inp,
        )
    name_matched = name_in_text(actual_name, debtor_text)
    dob_matched = actual_dob in dates
    inp |= {"name_matched": name_matched, "dob_matched": dob_matched}
    if name_matched and dob_matched:
        return _allow(Rule.IDENTITY_VERIFICATION, now, "identity verified", inp)
    codes = ["identity_mismatch"]
    reasons = ["name or date of birth did not match"]
    if failed_attempts + 1 >= max_failed:
        codes.append("max_attempts_reached")
        reasons.append("no verification attempts left")
    return _block(Rule.IDENTITY_VERIFICATION, now, codes, reasons, inp)
