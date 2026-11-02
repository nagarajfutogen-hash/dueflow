"""Admin dashboard D-2 (PLANNING §16): customers, payment calendar, needs attention.

Read-only, in plain words. Shown: test customers, plus any customer with a call made by hand
(machine refs that only evaluations used are left out). Dates of birth are never returned
and phone numbers are masked: the business view does not need them.
"""

from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from app.admin import (
    BUSINESS_TZ,
    CALLBACK_REASON_WORDS,
    HANDOFF_WORDS,
    LIVE_WINDOW,
    call_status,
    calls,
    result_words,
    safe_transcript,
    schedule_total,
)
from app.agent.scripts import spoken_date
from app.calls import MACHINE_REF_PREFIXES
from app.db.models import (
    Callback,
    CallbackStatus,
    ContactPrefs,
    Conversation,
    Debtor,
    Dispute,
    DisputeStatus,
    Event,
    Handoff,
    Payment,
    Promise,
    PromiseKind,
    PromiseStatus,
)
from app.policy.schedule import add_months

ATTENTION_DAYS = 30
CALENDAR_MAX_DAYS = 92
DATES_FROM_CALLS = 20  # latest calls scanned for dates the customer mentioned

PROMISE_WORDS = {
    PromiseStatus.ACTIVE: "Active",
    PromiseStatus.KEPT: "Kept",
    PromiseStatus.BROKEN: "Broken",
    PromiseStatus.SUPERSEDED: "Replaced by a newer plan",
}
CALLBACK_WORDS = {
    CallbackStatus.REQUESTED: "Waiting for your approval",
    CallbackStatus.APPROVED: "Approved",
    CallbackStatus.REJECTED: "Rejected",
    CallbackStatus.DONE: "Done",
}


def mask_phone(number: str | None) -> str | None:
    """Only the last 3 digits: enough to tell numbers apart, not to dial them."""
    if not number:
        return None
    digits = [c for c in number if c.isdigit()]
    return "•••" + "".join(digits[-3:]) if digits else "•••"


async def _shown(conn: AsyncConnection) -> set[int]:
    """Debtor ids the business view shows: test customers, and anyone called by hand."""
    tester = {
        debtor_id
        for debtor_id, ref in await conn.execute(select(Debtor.id, Debtor.external_ref))
        if not ref.startswith(MACHINE_REF_PREFIXES)
    }
    by_hand = set(
        await conn.scalars(
            select(Conversation.debtor_id).where(Conversation.scenario_id.is_(None)).distinct()
        )
    )
    return tester | by_hand


def _payment_words(kind: str, index: int, months: int) -> str:
    if kind == "down_payment":
        return "Down payment"
    if kind == "full_payment":
        return "Full payment"
    return f"Monthly payment {index} of {months}"


def promise_schedule(
    months: int,
    installment_jpy: int,
    first_due: date,
    kind: PromiseKind,
    terms_schedule: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], bool]:
    """Every payment of a plan, and whether the schedule is complete. Plans agreed by the
    agent store their full schedule; older seeded plans only have the monthly payments (their
    down payment date was never stored), so those are shown and flagged as partial."""
    if terms_schedule:
        items = list(terms_schedule)
        partial = False
    else:
        item_kind = "full_payment" if kind is PromiseKind.PAY_IN_FULL else "installment"
        items = [
            {
                "due": add_months(first_due, i).isoformat(),
                "amount_jpy": installment_jpy,
                "kind": item_kind,
            }
            for i in range(months)
        ]
        partial = kind is not PromiseKind.PAY_IN_FULL
    monthly = sum(1 for p in items if p["kind"] == "installment")
    out, index = [], 0
    for p in items:
        index += p["kind"] == "installment"
        out.append(
            {
                "due": p["due"],
                "due_words": spoken_date(date.fromisoformat(p["due"])),
                "amount_jpy": int(p["amount_jpy"]),
                "what": _payment_words(p["kind"], index, monthly),
            }
        )
    return out, partial


async def _terms_schedules(conn: AsyncConnection, event_ids: list[int]) -> dict[int, Any]:
    if not event_ids:
        return {}
    rows = await conn.execute(
        select(Event.id, Event.input["schedule"]).where(Event.id.in_(event_ids))
    )
    return {r[0]: r[1] for r in rows}


# --- Customers --------------------------------------------------------------------------------


async def customers(conn: AsyncConnection, *, now: datetime) -> list[dict[str, Any]]:
    shown = await _shown(conn)
    rows = [
        r
        for r in await AsyncSession(bind=conn).execute(
            select(Debtor, ContactPrefs)
            .outerjoin(ContactPrefs, ContactPrefs.debtor_id == Debtor.id)
            .order_by(Debtor.external_ref)
        )
        if r[0].id in shown
    ]
    disputed = set(
        await conn.scalars(select(Dispute.debtor_id).where(Dispute.status == DisputeStatus.OPEN))
    )
    active_plan = set(
        await conn.scalars(select(Promise.debtor_id).where(Promise.status == PromiseStatus.ACTIVE))
    )
    waiting = set(
        await conn.scalars(
            select(Callback.debtor_id).where(Callback.status == CallbackStatus.REQUESTED)
        )
    )
    approved = set(
        await conn.scalars(
            select(Callback.debtor_id).where(Callback.status == CallbackStatus.APPROVED)
        )
    )
    counts = {
        debtor_id: n
        for debtor_id, n in await conn.execute(
            select(Conversation.debtor_id, func.count()).group_by(Conversation.debtor_id)
        )
    }
    last_ids = select(func.max(Conversation.id)).group_by(Conversation.debtor_id)
    last = {
        r.debtor_id: r
        for r in await conn.execute(
            select(
                Conversation.debtor_id,
                Conversation.id,
                Conversation.created_at,
                Conversation.outcome,
            ).where(Conversation.id.in_(last_ids))
        )
    }
    out = []
    for debtor, prefs in rows:
        badges = []
        if debtor.id in disputed:
            badges.append("Open dispute")
        if prefs and prefs.stop_contact:
            badges.append("Don't call")
        if prefs and prefs.no_workplace_contact:
            badges.append("Not at work")
        if debtor.id in active_plan:
            badges.append("Active plan")
        if debtor.id in waiting:
            badges.append("Callback waiting")
        if debtor.id in approved:
            badges.append("Callback approved")
        lc = last.get(debtor.id)
        status = call_status(lc.outcome, lc.created_at, now) if lc else None
        out.append(
            {
                "ref": debtor.external_ref,
                "name": debtor.full_name,
                "balance_jpy": debtor.balance_jpy,
                "timezone": debtor.timezone,
                "badges": badges,
                "calls": int(counts.get(debtor.id, 0)),
                "last_call": None
                if lc is None or status is None
                else {
                    "id": lc.id,
                    "at": lc.created_at.isoformat(),
                    "result_words": result_words(lc.outcome, status),
                    "result": lc.outcome.value if lc.outcome else status,
                },
            }
        )
    return out


async def customer(conn: AsyncConnection, ref: str, *, now: datetime) -> dict[str, Any] | None:
    row = (
        await AsyncSession(bind=conn).execute(
            select(Debtor, ContactPrefs)
            .outerjoin(ContactPrefs, ContactPrefs.debtor_id == Debtor.id)
            .where(Debtor.external_ref == ref)
        )
    ).one_or_none()
    if row is None or row[0].id not in await _shown(conn):
        return None
    debtor, prefs = row
    payments = list(
        await conn.execute(
            select(Payment.amount_jpy, Payment.paid_at, Payment.method, Payment.reference)
            .where(Payment.debtor_id == debtor.id)
            .order_by(Payment.paid_at.desc())
        )
    )
    promises = list(
        await AsyncSession(bind=conn).execute(
            select(Promise)
            .where(Promise.debtor_id == debtor.id)
            .order_by(Promise.created_at.desc(), Promise.id.desc())
        )
    )
    schedules = await _terms_schedules(
        conn, [p.terms_event_id for (p,) in promises if p.terms_event_id]
    )
    plans = []
    for (p,) in promises:
        items, partial = promise_schedule(
            p.months,
            p.installment_jpy,
            p.first_due_date,
            p.kind,
            schedules.get(p.terms_event_id) if p.terms_event_id else None,
        )
        plans.append(
            {
                "status": p.status.value,
                "status_words": PROMISE_WORDS[p.status],
                "kind": p.kind.value,
                "made_on": p.created_at.isoformat(),
                "total_jpy": schedule_total(items) if not partial else None,
                "payments": items,
                "partial_schedule": partial,
            }
        )
    disputes = list(
        await conn.execute(
            select(Dispute.status, Dispute.reason, Dispute.opened_at, Dispute.resolved_at)
            .where(Dispute.debtor_id == debtor.id)
            .order_by(Dispute.opened_at.desc())
        )
    )
    callbacks = list(
        await conn.execute(
            select(
                Callback.id,
                Callback.due_date,
                Callback.status,
                Callback.reason,
                Callback.customer_words,
                Callback.conversation_id,
            )
            .where(Callback.debtor_id == debtor.id)
            .order_by(Callback.due_date.desc(), Callback.id.desc())
        )
    )
    timeline = await calls(conn, status="all", source="all", now=now, debtor_id=debtor.id)
    mentioned = []
    for c in timeline[:DATES_FROM_CALLS]:
        _, dates = await safe_transcript(conn, c["id"], debtor.date_of_birth)
        mentioned += [{**d, "call": c["id"], "call_at": c["started_at"]} for d in dates]
    return {
        "ref": debtor.external_ref,
        "name": debtor.full_name,
        "timezone": debtor.timezone,
        "balance_jpy": debtor.balance_jpy,
        "paid_total_jpy": sum(p.amount_jpy for p in payments),
        "plan_limits": {
            "min_monthly_jpy": debtor.min_monthly_jpy,
            "max_months": debtor.max_months,
        },
        "phones": {
            "mobile": mask_phone(debtor.phone_mobile),
            "home": mask_phone(debtor.phone_home),
            "work": mask_phone(debtor.phone_work),
        },
        "contact": {
            "stop_contact": bool(prefs and prefs.stop_contact),
            "no_workplace_contact": bool(prefs and prefs.no_workplace_contact),
        },
        "notes": debtor.notes,
        "payments": [
            {
                "amount_jpy": p.amount_jpy,
                "paid_on": p.paid_at.astimezone(BUSINESS_TZ).date().isoformat(),
                "method": p.method,
                "reference": p.reference,
            }
            for p in payments
        ],
        "plans": plans,
        "disputes": [
            {
                "open": d.status is DisputeStatus.OPEN,
                "reason": d.reason,
                "opened_on": d.opened_at.date().isoformat(),
                "resolved_on": d.resolved_at.date().isoformat() if d.resolved_at else None,
            }
            for d in disputes
        ],
        "callbacks": [
            {
                "id": c.id,
                "due_words": spoken_date(c.due_date),
                "status_words": CALLBACK_WORDS[c.status],
                "reason_words": CALLBACK_REASON_WORDS.get(c.reason, c.reason),
                "customer_words": c.customer_words,
                "requested_in_call": c.conversation_id,
            }
            for c in callbacks
        ],
        "calls": timeline,
        "dates_mentioned": mentioned,
    }


# --- Payment calendar -------------------------------------------------------------------------


async def calendar(
    conn: AsyncConnection, *, start: date, days: int, now: datetime
) -> list[dict[str, Any]]:
    """Payments due on active plans and callbacks, day by day. A passed due date with no
    payment on file since then says exactly that: nothing marks plans kept or broken."""
    days = max(1, min(days, CALENDAR_MAX_DAYS))
    end = start + timedelta(days=days - 1)
    today = now.astimezone(BUSINESS_TZ).date()
    by_day: dict[date, list[dict[str, Any]]] = defaultdict(list)
    shown = await _shown(conn)

    promises = [
        r
        for r in await AsyncSession(bind=conn).execute(
            select(Promise, Debtor.external_ref, Debtor.full_name)
            .join(Debtor, Debtor.id == Promise.debtor_id)
            .where(Promise.status == PromiseStatus.ACTIVE)
        )
        if r[0].debtor_id in shown
    ]
    schedules = await _terms_schedules(
        conn, [p.terms_event_id for p, _, _ in promises if p.terms_event_id]
    )
    last_paid = {
        debtor_id: paid_at
        for debtor_id, paid_at in await conn.execute(
            select(Payment.debtor_id, func.max(Payment.paid_at)).group_by(Payment.debtor_id)
        )
    }
    for p, ref, name in promises:
        items, partial = promise_schedule(
            p.months,
            p.installment_jpy,
            p.first_due_date,
            p.kind,
            schedules.get(p.terms_event_id) if p.terms_event_id else None,
        )
        paid_at = last_paid.get(p.debtor_id)
        for item in items:
            due = date.fromisoformat(item["due"])
            if not start <= due <= end:
                continue
            paid_since = paid_at is not None and paid_at.astimezone(BUSINESS_TZ).date() >= due
            by_day[due].append(
                {
                    "type": "payment",
                    "customer": name,
                    "customer_ref": ref,
                    "what": item["what"],
                    "amount_jpy": item["amount_jpy"],
                    "partial_schedule": partial,
                    "note": "Due date passed, no payment on file since"
                    if due < today and not paid_since
                    else None,
                }
            )
    for cb, ref, name in await AsyncSession(bind=conn).execute(
        select(Callback, Debtor.external_ref, Debtor.full_name)
        .join(Debtor, Debtor.id == Callback.debtor_id)
        .where(
            Callback.due_date.between(start, end),
            Callback.status != CallbackStatus.REJECTED,
        )
    ):
        if cb.debtor_id not in shown:
            continue
        by_day[cb.due_date].append(
            {
                "type": "callback",
                "customer": name,
                "customer_ref": ref,
                "what": f"Call back: {CALLBACK_REASON_WORDS.get(cb.reason, cb.reason)}",
                "amount_jpy": None,
                "partial_schedule": False,
                "note": CALLBACK_WORDS[cb.status],
            }
        )
    return [
        {
            "day": d.isoformat(),
            "day_words": spoken_date(d),
            "today": d == today,
            "items": sorted(items, key=lambda i: (i["type"], i["customer"])),
        }
        for d, items in sorted(by_day.items())
    ]


# --- Needs attention --------------------------------------------------------------------------


async def attention(conn: AsyncConnection, *, now: datetime) -> dict[str, Any]:
    since = now - timedelta(days=ATTENTION_DAYS)
    shown = await _shown(conn)
    handoffs = [
        {
            "customer": r.full_name,
            "customer_ref": r.external_ref,
            "reason_words": HANDOFF_WORDS.get(r.reason.value, r.reason.value),
            "note": r.summary,
            "call": r.conversation_id,
            "at": r.created_at.isoformat(),
        }
        for r in await conn.execute(
            select(
                Handoff.reason,
                Handoff.summary,
                Handoff.conversation_id,
                Handoff.created_at,
                Debtor.id.label("debtor_id"),
                Debtor.full_name,
                Debtor.external_ref,
            )
            .join(Debtor, Debtor.id == Handoff.debtor_id)
            .where(Handoff.created_at >= since)
            .order_by(Handoff.created_at.desc())
        )
        if r.debtor_id in shown
    ]
    disputes = [
        {
            "customer": r.full_name,
            "customer_ref": r.external_ref,
            "reason": r.reason,
            "opened_on": r.opened_at.date().isoformat(),
        }
        for r in await conn.execute(
            select(
                Debtor.id.label("debtor_id"),
                Dispute.reason,
                Dispute.opened_at,
                Debtor.full_name,
                Debtor.external_ref,
            )
            .join(Debtor, Debtor.id == Dispute.debtor_id)
            .where(Dispute.status == DisputeStatus.OPEN)
            .order_by(Dispute.opened_at.desc())
        )
        if r.debtor_id in shown
    ]
    stop = [
        {
            "customer": r.full_name,
            "customer_ref": r.external_ref,
            "at": r.stop_contact_at.isoformat() if r.stop_contact_at else None,
        }
        for r in await conn.execute(
            select(
                Debtor.id.label("debtor_id"),
                ContactPrefs.stop_contact_at,
                Debtor.full_name,
                Debtor.external_ref,
            )
            .join(Debtor, Debtor.id == ContactPrefs.debtor_id)
            .where(ContactPrefs.stop_contact.is_(True), ContactPrefs.stop_contact_at >= since)
            .order_by(ContactPrefs.stop_contact_at.desc())
        )
        if r.debtor_id in shown
    ]
    unfinished = [
        {
            "customer": r.full_name,
            "customer_ref": r.external_ref,
            "call": r.id,
            "at": r.created_at.isoformat(),
        }
        for r in await conn.execute(
            select(
                Debtor.id.label("debtor_id"),
                Conversation.id,
                Conversation.created_at,
                Debtor.full_name,
                Debtor.external_ref,
            )
            .join(Debtor, Debtor.id == Conversation.debtor_id)
            .where(
                Conversation.outcome.is_(None),
                Conversation.scenario_id.is_(None),
                Conversation.created_at >= since,
                Conversation.created_at <= now - LIVE_WINDOW,
            )
            .order_by(Conversation.created_at.desc())
        )
        if r.debtor_id in shown
    ]
    return {
        "days": ATTENTION_DAYS,
        "sent_to_team": handoffs,
        "open_disputes": disputes,
        "stop_contact": stop,
        "unfinished_calls": unfinished,
    }
