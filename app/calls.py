"""Simulated outbound dial: PreCallCheck runs before any conversation exists.

Blocked dials create no conversation (their events carry only debtor_id). Allowed dials
create the conversation first, so the pre-call events are linked to it.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import func, insert, select, true, update
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from app.db.audit import record_decisions
from app.db.models import (
    Callback,
    CallbackStatus,
    Channel,
    ContactPrefs,
    Conversation,
    Debtor,
    Dispute,
    DisputeStatus,
    Payment,
    PhoneType,
    Promise,
    PromiseStatus,
    ResultLabel,
)
from app.policy.config import PolicyRules
from app.policy.engine import Decision, DialTarget, pre_call_check


class DebtorNotFoundError(LookupError):
    pass


class NoNumberForTargetError(ValueError):
    pass


@dataclass(frozen=True)
class DialRequest:
    debtor_ref: str
    sim_time: datetime
    target: DialTarget
    agent_version: str
    label: ResultLabel = ResultLabel.SYNTHETIC
    channel: Channel = Channel.TEXT
    scenario_id: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class DialResult:
    allowed: bool
    conversation_id: int | None
    decisions: list[Decision]
    event_ids: list[int]


def _number_for(debtor: Debtor, target: DialTarget) -> str | None:
    return {
        DialTarget.HOME: debtor.phone_home,
        DialTarget.MOBILE: debtor.phone_mobile,
        DialTarget.WORK: debtor.phone_work,
    }[target]


async def account_snapshot(orm: AsyncSession, debtor: Debtor) -> dict[str, Any]:
    """Account facts at dial time (for replay). No name, DOB or phone numbers."""
    conn = await orm.connection()
    promises = await orm.scalars(
        select(Promise).where(Promise.debtor_id == debtor.id).order_by(Promise.id)
    )
    open_disputes = (
        await conn.execute(
            select(Dispute.id).where(
                Dispute.debtor_id == debtor.id, Dispute.status == DisputeStatus.OPEN
            )
        )
    ).scalars()
    paid = (
        await conn.execute(select(Payment.amount_jpy).where(Payment.debtor_id == debtor.id))
    ).scalars()
    promise_list = [
        {
            "id": p.id,
            "status": p.status.value,
            "months": p.months,
            "down_payment_jpy": p.down_payment_jpy,
            "installment_jpy": p.installment_jpy,
            "first_due_date": p.first_due_date.isoformat(),
        }
        for p in promises
    ]
    return {
        "debtor_ref": debtor.external_ref,
        "timezone": debtor.timezone,
        "balance_jpy": debtor.balance_jpy,
        "has_open_dispute": any(True for _ in open_disputes),
        "has_broken_promise": any(p["status"] == PromiseStatus.BROKEN for p in promise_list),
        "promises": promise_list,
        "payments_total_jpy": sum(paid),
        # Agent v5: this customer's own plan limits (None = global rule), for replays.
        "customer_limits": {
            "min_monthly_jpy": debtor.min_monthly_jpy,
            "max_months": debtor.max_months,
        },
    }


async def dial(conn: AsyncConnection, req: DialRequest, rules: PolicyRules) -> DialResult:
    # ORM reads share this connection's transaction.
    orm = AsyncSession(bind=conn)
    debtor = await orm.scalar(select(Debtor).where(Debtor.external_ref == req.debtor_ref))
    if debtor is None:
        raise DebtorNotFoundError(req.debtor_ref)
    if _number_for(debtor, req.target) is None:
        raise NoNumberForTargetError(f"debtor has no {req.target.value} number")

    prefs = await orm.scalar(select(ContactPrefs).where(ContactPrefs.debtor_id == debtor.id))
    # Every earlier allowed dial is an attempt (blocked dials never reached the debtor).
    prior_attempts = list(
        (
            await conn.execute(
                select(Conversation.sim_started_at).where(Conversation.debtor_id == debtor.id)
            )
        ).scalars()
    )
    decisions = pre_call_check(
        req.sim_time,
        debtor_tz=debtor.timezone,
        prior_attempts=prior_attempts,
        stop_contact=prefs.stop_contact if prefs else False,
        target=req.target,
        no_workplace_contact=prefs.no_workplace_contact if prefs else False,
        workplace_justification=prefs.workplace_justification if prefs else None,
        rules=rules,
    )
    allowed = all(d.allowed for d in decisions)

    conversation_id: int | None = None
    if allowed:
        conversation_id = (
            await conn.execute(
                insert(Conversation)
                .values(
                    debtor_id=debtor.id,
                    channel=req.channel,
                    label=req.label,
                    agent_version=req.agent_version,
                    scenario_id=req.scenario_id,
                    seed=req.seed,
                    sim_started_at=req.sim_time,
                    dialed_phone_type=PhoneType(req.target.value),
                    account_snapshot=await account_snapshot(orm, debtor),
                )
                .returning(Conversation.id)
            )
        ).scalar_one()

    event_ids = await record_decisions(
        conn,
        decisions,
        agent_version=req.agent_version,
        conversation_id=conversation_id,
        debtor_id=debtor.id,
    )
    return DialResult(allowed, conversation_id, decisions, event_ids)


# --- Auto dial (dev page): pick a debtor, time and number that should pass the check -----

AUTO_LOCAL_TIME = time(12, 0)  # noon, debtor local time: inside the contact window
# Debtors made per conversation by eval runs and live tests: never offered to a tester.
MACHINE_REF_PREFIXES = ("EV-", "LIVE-")


def is_tester_debtor(ref: str) -> bool:
    return not ref.startswith(MACHINE_REF_PREFIXES)


@dataclass(frozen=True)
class AutoPick:
    debtor_ref: str
    sim_time: datetime
    target: DialTarget
    # v5: the owner-approved callback this dial fulfils (marked done once allowed).
    callback_id: int | None = None


def auto_target(debtor: Debtor, prefs: ContactPrefs | None) -> DialTarget | None:
    """Mobile, then home; work only when allowed and justified (else the check blocks it)."""
    if debtor.phone_mobile:
        return DialTarget.MOBILE
    if debtor.phone_home:
        return DialTarget.HOME
    if (
        debtor.phone_work
        and prefs
        and not prefs.no_workplace_contact
        and prefs.workplace_justification
    ):
        return DialTarget.WORK
    return None


def auto_sim_time(
    prior_attempts: list[datetime], debtor_tz: str, rules: PolicyRules, today: date
) -> datetime:
    """Noon local on the first day (from today, after the last attempt) under both attempt
    limits. Only a guess for picking: the real dial still runs and records the full check."""
    zone = ZoneInfo(debtor_tz)
    day = today
    if prior_attempts:
        day = max(day, max(prior_attempts).astimezone(zone).date() + timedelta(days=1))
    limits = rules.attempt_limits
    while True:
        at = datetime.combine(day, AUTO_LOCAL_TIME, tzinfo=zone)
        same_day = sum(1 for a in prior_attempts if a.astimezone(zone).date() == day)
        week = sum(1 for a in prior_attempts if at - timedelta(days=7) < a <= at)
        if same_day < limits.per_local_day and week < limits.per_rolling_7_days:
            return at
        day += timedelta(days=1)


def _any_target(debtor: Debtor) -> DialTarget | None:
    return next((t for t in DialTarget if _number_for(debtor, t)), None)


async def auto_picks(
    conn: AsyncConnection, rules: PolicyRules, today: date, debtor_ref: str | None = None
) -> list[AutoPick]:
    """Callable-looking debtors, least recently dialed first (never-dialed before all);
    stop-contact and open-dispute debtors are left out.
    v5 callbacks: owner-approved callbacks that are due come first (noon local on the due
    date or later); a debtor whose callback still waits for approval, or is approved for a
    later day, is left out until then. The pre-call check still decides every dial.
    With debtor_ref: only that debtor, never skipped, so a stop-contact or work-only block
    is still dialed and shown (the time and number are still the best available)."""
    last_dial = (
        select(Conversation.debtor_id, func.max(Conversation.id).label("last_id"))
        .group_by(Conversation.debtor_id)
        .subquery()
    )
    orm = AsyncSession(bind=conn)
    rows = (
        await orm.execute(
            select(Debtor, ContactPrefs)
            .outerjoin(ContactPrefs, ContactPrefs.debtor_id == Debtor.id)
            .outerjoin(last_dial, last_dial.c.debtor_id == Debtor.id)
            .where(true() if debtor_ref is None else Debtor.external_ref == debtor_ref)
            .order_by(last_dial.c.last_id.asc().nulls_first(), Debtor.id)
        )
    ).all()
    disputed = set(
        (
            await conn.execute(
                select(Dispute.debtor_id).where(Dispute.status == DisputeStatus.OPEN)
            )
        ).scalars()
    )
    open_callbacks = (
        await conn.execute(
            select(Callback.id, Callback.debtor_id, Callback.status, Callback.due_date)
            .where(Callback.status.in_((CallbackStatus.REQUESTED, CallbackStatus.APPROVED)))
            .order_by(Callback.due_date, Callback.id)
        )
    ).all()
    due: dict[int, tuple[int, date]] = {}
    waiting: set[int] = set()
    for cb in open_callbacks:
        if cb.status is CallbackStatus.APPROVED and cb.due_date <= today:
            due.setdefault(cb.debtor_id, (cb.id, cb.due_date))
        else:
            waiting.add(cb.debtor_id)
    picks: list[AutoPick] = []
    for debtor, prefs in rows:
        callback = due.get(debtor.id)
        target = auto_target(debtor, prefs)
        if debtor_ref is not None:
            target = target or _any_target(debtor)
        elif not is_tester_debtor(debtor.external_ref) or (prefs and prefs.stop_contact):
            continue
        elif debtor.id in disputed:
            continue  # an open dispute ends the call at once: not a useful auto pick
        elif callback is None and debtor.id in waiting:
            continue  # v5: never called before the owner approves the callback, or before it
        if target is None:
            continue
        prior = list(
            (
                await conn.execute(
                    select(Conversation.sim_started_at).where(Conversation.debtor_id == debtor.id)
                )
            ).scalars()
        )
        start = max(today, callback[1]) if callback else today
        picks.append(
            AutoPick(
                debtor.external_ref,
                auto_sim_time(prior, debtor.timezone, rules, start),
                target,
                callback_id=callback[0] if callback else None,
            )
        )
    picks.sort(key=lambda p: p.callback_id is None)  # due callbacks first (stable otherwise)
    return picks


async def auto_dial(
    conn: AsyncConnection,
    rules: PolicyRules,
    *,
    today: date,
    agent_version: str,
    label: ResultLabel,
    channel: Channel = Channel.TEXT,
    debtor_ref: str | None = None,
    max_tries: int = 5,
) -> tuple[AutoPick, DialResult] | None:
    """Dial the first pick the pre-call check allows; None if none of the first max_tries
    is allowed. A blocked try is recorded like any blocked dial (so callers commit).
    With debtor_ref: that debtor is dialed once and the result returned, allowed or not."""
    picks = await auto_picks(conn, rules, today, debtor_ref)
    if debtor_ref is not None and not picks:
        raise DebtorNotFoundError(debtor_ref)
    for pick in picks[:max_tries]:
        req = DialRequest(
            debtor_ref=pick.debtor_ref,
            sim_time=pick.sim_time,
            target=pick.target,
            agent_version=agent_version,
            label=label,
            channel=channel,
        )
        result = await dial(conn, req, rules)
        if result.allowed and pick.callback_id is not None:
            await conn.execute(
                update(Callback)
                .where(Callback.id == pick.callback_id)
                .values(status=CallbackStatus.DONE, done_conversation_id=result.conversation_id)
            )
        if result.allowed or debtor_ref is not None:
            return pick, result
    return None
