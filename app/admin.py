"""Admin dashboard queries (PLANNING §16): read-only, in plain words for business users.

Every number comes from stored rows and carries its label (SYNTHETIC / HUMAN TEST). Call
summaries are built here from events, promises and handoffs; no LLM writes them. Dates of
birth never leave this module: identity turns are hidden and a date equal to the debtor's
date of birth is never listed as a mentioned date.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from sqlalchemy import ColumnElement, func, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from app.agent.scripts import VERIFIED, VERIFIED_WARM, spoken_date, yen
from app.db.audit import record_tool_call
from app.db.models import (
    Callback,
    CallbackStatus,
    Conversation,
    Debtor,
    Decision,
    Event,
    EventType,
    Handoff,
    Outcome,
    Promise,
    PromiseStatus,
    Speaker,
    Turn,
)
from app.policy.identity import find_dates

# The business day for "today" and "due this week". Debtor-side rules use each debtor's own
# timezone; this is only how the dashboard groups real call times.
BUSINESS_TZ = ZoneInfo("Asia/Tokyo")
# A call with no result yet counts as live this long after it was dialed; after that it is
# shown as "not finished" (the browser closed without an ending).
LIVE_WINDOW = timedelta(minutes=30)
MAX_CALLS = 200
DUE_SOON_DAYS = 7

Source = Literal["manual", "eval", "all"]
Period = Literal["today", "7d", "30d", "all"]
Status = Literal["live", "all"]

RESULT_WORDS: dict[str, str] = {
    Outcome.PROMISE.value: "Agreed to pay",
    Outcome.HANDOFF.value: "Sent to team",
    Outcome.STOP_CONTACT.value: "Asked not to be called",
    Outcome.END_WRONG_PERSON.value: "Not the right person",
    Outcome.THIRD_PARTY_CALLBACK.value: "Left a callback message",
    Outcome.NO_VERIFICATION.value: "Couldn't confirm identity",
    Outcome.REFUSE_DISCOUNT.value: "Discount refused",
    Outcome.NO_RESPONSE.value: "No answer",
    Outcome.DISCONNECTED.value: "Call dropped",
    Outcome.CALLBACK_SCHEDULED.value: "Callback requested",
}
STATUS_WORDS = {"live": "In progress", "ended": "Finished", "unfinished": "Not finished"}
HANDOFF_WORDS = {
    "hardship": "can't afford any allowed plan",
    "dispute": "disputes the debt or says it is paid",
    "unresolved": "needs a person to follow up",
    "error": "a system problem",
    "below_account_floor": "can't afford the lowest allowed plan",
}
TERMS_RULES = ("plan_terms", "pay_in_full")
_VERIFIED_LINES = (VERIFIED, VERIFIED_WARM)
HIDDEN_IDENTITY_TEXT = "(Gave their name and date of birth: hidden)"


def _period_start(period: Period, now: datetime) -> datetime | None:
    if period == "today":
        local = now.astimezone(BUSINESS_TZ)
        return local.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "7d":
        return now - timedelta(days=7)
    if period == "30d":
        return now - timedelta(days=30)
    return None


def _source_filter(source: Source) -> ColumnElement[bool] | None:
    """manual = calls made by hand from the dev page; eval = simulated evaluation runs."""
    if source == "manual":
        return Conversation.scenario_id.is_(None)
    if source == "eval":
        return Conversation.scenario_id.is_not(None)
    return None


def call_status(outcome: Outcome | None, created_at: datetime, now: datetime) -> str:
    if outcome is not None:
        return "ended"
    return "live" if now - created_at < LIVE_WINDOW else "unfinished"


def result_words(outcome: Outcome | None, status: str) -> str:
    if outcome is None:
        return STATUS_WORDS[status]
    return RESULT_WORDS.get(outcome.value, outcome.value)


def schedule_words(schedule: list[dict[str, Any]]) -> str:
    """'¥15,000 down by November 7th, 2026, then 5 monthly payments of ¥27,000'."""
    first = schedule[0]
    if first["kind"] == "full_payment":
        return f"pay {yen(first['amount_jpy'])} in full by {spoken_date(_d(first['due']))}"
    installments = schedule[1:]
    text = (
        f"{yen(first['amount_jpy'])} down by {spoken_date(_d(first['due']))}, then "
        f"{len(installments)} monthly payments of {yen(installments[0]['amount_jpy'])}"
    )
    if installments[-1]["amount_jpy"] != installments[0]["amount_jpy"]:
        text += f" (the last one {yen(installments[-1]['amount_jpy'])})"
    return text


def _d(raw: str) -> date:
    return date.fromisoformat(raw)


def schedule_total(schedule: list[dict[str, Any]]) -> int:
    return sum(int(p["amount_jpy"]) for p in schedule)


@dataclass
class CallFacts:
    """What happened in one call, read from its events, promise and handoff."""

    verified: bool = False
    id_mismatches: int = 0
    offers: int = 0
    plans_refused: int = 0
    discount_refused: bool = False
    payment_claims: list[bool] = field(default_factory=list)  # True = matched a payment
    approach: str | None = None
    no_workplace: bool = False
    promise_schedule: list[dict[str, Any]] | None = None
    handoff_reason: str | None = None
    ended_at: datetime | None = None
    callback_due: date | None = None  # v5: the day the customer asked to be called
    moments: list[dict[str, Any]] = field(default_factory=list)


def summarize(outcome: Outcome | None, status: str, f: CallFacts) -> str:
    """One or two plain sentences for a manager, built only from stored facts."""
    parts: list[str] = []
    if f.verified:
        parts.append("Identity confirmed.")
    elif f.id_mismatches:
        parts.append("Identity details didn't match.")
    if f.payment_claims:
        found = any(f.payment_claims)
        parts.append(
            "Said they already paid: "
            + ("a matching payment is on file." if found else "no matching payment found.")
        )
    if f.discount_refused:
        parts.append("Asked for a discount (not allowed).")
    if f.offers and f.promise_schedule is None:
        parts.append(f"{f.offers} payment option{'s' if f.offers != 1 else ''} offered.")
    if f.promise_schedule is not None:
        parts.append(f"Agreed to {schedule_words(f.promise_schedule)}.")
    if f.handoff_reason is not None:
        parts.append(f"Sent to team: {HANDOFF_WORDS.get(f.handoff_reason, f.handoff_reason)}.")
    if f.callback_due is not None:
        parts.append(f"Asked for a call back on {spoken_date(f.callback_due)}.")
    if f.no_workplace:
        parts.append("Asked not to be called at work.")
    if outcome is Outcome.STOP_CONTACT:
        parts.append("Asked not to be called again.")
    if outcome is None:
        parts.append("Call in progress." if status == "live" else "The call has no ending.")
    elif outcome is not Outcome.STOP_CONTACT and not (
        f.promise_schedule or f.handoff_reason or f.callback_due
    ):
        parts.append(f"{result_words(outcome, status)}.")
    return " ".join(parts)


def _moment(at: datetime | None, kind: str, text: str) -> dict[str, Any]:
    return {"at": at.isoformat() if at else None, "kind": kind, "text": text}


def _apply_event(f: CallFacts, e: Any, schedules: dict[int, list[dict[str, Any]]]) -> None:
    allowed = e.decision == Decision.ALLOWED
    codes = (e.input or {}).get("codes", [])
    at = e.clock_time
    if e.type == EventType.TOOL_CALL:
        tool = (e.input or {}).get("tool")
        if tool == "set_approach":
            f.approach = e.input.get("approach")
            f.moments.append(_moment(at, "tone", f"Agent chose a {f.approach} tone"))
        elif tool == "record_no_workplace_contact":
            f.no_workplace = True
            f.moments.append(_moment(at, "contact", "Asked not to be called at work"))
        elif tool == "record_stop_contact":
            f.moments.append(_moment(at, "contact", "Asked not to be called again"))
        return
    if e.type == EventType.STATE_TRANSITION:
        if (e.input or {}).get("to") == "End":
            f.ended_at = at  # the "call ended" moment is added last, in plain words
        return
    if e.rule == "identity_verification":
        if allowed:
            f.verified = True
            f.moments.append(_moment(at, "identity", "Identity confirmed"))
        elif "identity_mismatch" in codes:
            f.id_mismatches += 1
            f.moments.append(_moment(at, "identity", "Identity details didn't match"))
    elif e.rule in TERMS_RULES:
        if allowed:
            f.offers += 1
            schedule = (e.input or {}).get("schedule")
            if schedule:
                schedules[e.id] = schedule
                f.moments.append(_moment(at, "offer", f"Offered: {schedule_words(schedule)}"))
        else:
            f.plans_refused += 1
            f.moments.append(_moment(at, "offer", "A plan they asked for was outside the rules"))
    elif e.rule == "callback_request":
        if allowed:
            f.callback_due = _d(e.input["requested"])
            f.moments.append(
                _moment(at, "callback", f"Asked for a call back on {spoken_date(f.callback_due)}")
            )
        else:
            f.moments.append(_moment(at, "callback", "A call-back day could not be booked"))
    elif e.rule == "negotiation_ladder" and not allowed:
        f.moments.append(_moment(at, "handoff", "No allowed plan fits: handed to the team"))
    elif e.rule == "no_discount" and not allowed:
        f.discount_refused = True
        f.moments.append(_moment(at, "discount", "Asked for a discount: not allowed"))
    elif e.rule == "payment_record":
        f.payment_claims.append(allowed)
        f.moments.append(
            _moment(
                at,
                "payment",
                "Said they already paid: "
                + ("matches a payment on file" if allowed else "no matching payment on file"),
            )
        )


async def call_facts(conn: AsyncConnection, ids: list[int]) -> dict[int, CallFacts]:
    """Facts for many calls in three queries (events, promises, handoffs)."""
    facts = {cid: CallFacts() for cid in ids}
    if not ids:
        return facts
    events = await conn.execute(
        select(
            Event.id,
            Event.conversation_id,
            Event.type,
            Event.rule,
            Event.decision,
            Event.reason,
            Event.input,
            Event.clock_time,
        )
        .where(
            Event.conversation_id.in_(ids),
            Event.type.in_(
                (EventType.POLICY_DECISION, EventType.TOOL_CALL, EventType.STATE_TRANSITION)
            ),
        )
        .order_by(Event.id)
    )
    schedules: dict[int, list[dict[str, Any]]] = {}
    for e in events:
        _apply_event(facts[e.conversation_id], e, schedules)
    promises = await conn.execute(
        select(Promise.conversation_id, Promise.terms_event_id, Promise.created_at).where(
            Promise.conversation_id.in_(ids)
        )
    )
    for p in promises:
        f = facts[p.conversation_id]
        f.promise_schedule = schedules.get(p.terms_event_id) if p.terms_event_id else None
        if f.promise_schedule is not None:
            f.moments.append(
                _moment(None, "promise", f"Agreed to {schedule_words(f.promise_schedule)}")
            )
    handoffs = await conn.execute(
        select(Handoff.conversation_id, Handoff.reason, Handoff.summary).where(
            Handoff.conversation_id.in_(ids)
        )
    )
    for h in handoffs:
        f = facts[h.conversation_id]
        f.handoff_reason = h.reason.value
        f.moments.append(
            _moment(
                None,
                "handoff",
                f"Sent to team ({HANDOFF_WORDS.get(f.handoff_reason, f.handoff_reason)}). "
                f"Agent's note: {h.summary}",
            )
        )
    return facts


def _row(c: Any, now: datetime, f: CallFacts) -> dict[str, Any]:
    status = call_status(c.outcome, c.created_at, now)
    return {
        "id": c.id,
        "customer": c.full_name,
        "customer_ref": c.external_ref,
        "started_at": c.created_at.isoformat(),
        "sim_started_at": c.sim_started_at.isoformat(),
        "channel": c.channel.value,
        "label": c.label.value,
        "agent_version": c.agent_version,
        "source": "manual" if c.scenario_id is None else "eval",
        "status": status,
        "result": c.outcome.value if c.outcome else None,
        "result_words": result_words(c.outcome, status),
        "summary": summarize(c.outcome, status, f),
    }


_CALL_COLUMNS = (
    Conversation.id,
    Conversation.created_at,
    Conversation.sim_started_at,
    Conversation.channel,
    Conversation.label,
    Conversation.agent_version,
    Conversation.scenario_id,
    Conversation.outcome,
    Debtor.full_name,
    Debtor.external_ref,
)


async def calls(
    conn: AsyncConnection,
    *,
    status: Status,
    source: Source,
    now: datetime,
    limit: int = MAX_CALLS,
    debtor_id: int | None = None,
) -> list[dict[str, Any]]:
    stmt = select(*_CALL_COLUMNS).join(Debtor, Debtor.id == Conversation.debtor_id)
    if (cond := _source_filter(source)) is not None:
        stmt = stmt.where(cond)
    if debtor_id is not None:  # one customer's calls (customer page)
        stmt = stmt.where(Conversation.debtor_id == debtor_id)
    if status == "live":
        stmt = stmt.where(
            Conversation.outcome.is_(None), Conversation.created_at > now - LIVE_WINDOW
        )
    rows = list(await conn.execute(stmt.order_by(Conversation.id.desc()).limit(limit)))
    facts = await call_facts(conn, [r.id for r in rows])
    return [_row(r, now, facts[r.id]) for r in rows]


def _identity_turn_end(turns: list[Any]) -> int | None:
    """Seq of the 'identity confirmed' line; debtor turns before it are identity turns."""
    for t in turns:
        if t.speaker == Speaker.AGENT and t.text in _VERIFIED_LINES:
            return int(t.seq)
    return None


async def safe_transcript(
    conn: AsyncConnection, conversation_id: int, date_of_birth: date
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The transcript with identity turns hidden, and the dates the customer mentioned after
    verification. A turn holding the date of birth is always hidden and never listed."""
    turns = list(
        await conn.execute(
            select(Turn.seq, Turn.speaker, Turn.text, Turn.started_at)
            .where(Turn.conversation_id == conversation_id)
            .order_by(Turn.seq)
        )
    )
    verified_at = _identity_turn_end(turns)
    transcript: list[dict[str, Any]] = []
    mentioned: list[dict[str, Any]] = []
    for t in turns:
        text = t.text
        if t.speaker == Speaker.DEBTOR:
            dates = find_dates(text)
            before_verification = verified_at is None or t.seq < verified_at
            if dates and (before_verification or date_of_birth in dates):
                text = HIDDEN_IDENTITY_TEXT
            elif not before_verification:
                mentioned += [
                    {"date": d.isoformat(), "spoken": spoken_date(d), "turn": t.seq}
                    for d in sorted(dates)
                ]
        transcript.append(
            {
                "seq": t.seq,
                "who": "AI agent" if t.speaker == Speaker.AGENT else "Customer",
                "text": text,
                "at": t.started_at.isoformat(),
            }
        )
    return transcript, mentioned


async def call_detail(
    conn: AsyncConnection, conversation_id: int, now: datetime
) -> dict[str, Any] | None:
    c = (
        await conn.execute(
            select(*_CALL_COLUMNS, Debtor.date_of_birth, Conversation.account_snapshot)
            .join(Debtor, Debtor.id == Conversation.debtor_id)
            .where(Conversation.id == conversation_id)
        )
    ).one_or_none()
    if c is None:
        return None
    transcript, mentioned = await safe_transcript(conn, conversation_id, c.date_of_birth)
    f = (await call_facts(conn, [c.id]))[c.id]
    row = _row(c, now, f)
    if c.outcome is not None:
        f.moments.append(_moment(f.ended_at, "end", f"Call ended: {row['result_words']}"))
    snapshot = c.account_snapshot or {}
    return {
        "call": row,
        "balance_jpy": snapshot.get("balance_jpy"),
        "tone": f.approach,
        "transcript": transcript,
        "moments": f.moments,
        "dates_mentioned": mentioned,
    }


def _previous_window(
    period: Period, start: datetime | None, now: datetime
) -> tuple[datetime, datetime] | None:
    """The window just before this period, of the same length (yesterday for 'today')."""
    if start is None:
        return None
    length = timedelta(days=1) if period == "today" else now - start
    return start - length, start


async def _window(
    conn: AsyncConnection, *, source: Source, start: datetime | None, end: datetime | None = None
) -> tuple[list[Any], dict[int, CallFacts]]:
    stmt = select(
        Conversation.id, Conversation.label, Conversation.outcome, Conversation.created_at
    )
    if (cond := _source_filter(source)) is not None:
        stmt = stmt.where(cond)
    if start is not None:
        stmt = stmt.where(Conversation.created_at >= start)
    if end is not None:
        stmt = stmt.where(Conversation.created_at < end)
    rows = list(await conn.execute(stmt.order_by(Conversation.id)))
    return rows, await call_facts(conn, [r.id for r in rows])


def _headline(rows: list[Any], facts: dict[int, CallFacts]) -> dict[str, int]:
    promised = [f.promise_schedule for f in facts.values() if f.promise_schedule is not None]
    return {
        "calls": len(rows),
        "agreed_to_pay": len(promised),
        "sent_to_team": sum(f.handoff_reason is not None for f in facts.values()),
        "promised_jpy": sum(schedule_total(s) for s in promised),
    }


def _per_day(
    rows: list[Any], facts: dict[int, CallFacts], start: datetime | None, now: datetime
) -> list[dict[str, Any]]:
    """Calls per business day, split by result, with every day in the range (zeros included)."""
    if not rows:
        return []
    first = (start or rows[0].created_at).astimezone(BUSINESS_TZ).date()
    last = now.astimezone(BUSINESS_TZ).date()
    days = {
        first + timedelta(days=i): {"agreed": 0, "sent_to_team": 0, "other": 0}
        for i in range((last - first).days + 1)
    }
    for r in rows:
        day = days.get(r.created_at.astimezone(BUSINESS_TZ).date())
        if day is None:
            continue
        f = facts[r.id]
        key = (
            "agreed"
            if f.promise_schedule is not None
            else "sent_to_team"
            if f.handoff_reason is not None
            else "other"
        )
        day[key] += 1
    return [{"day": d.isoformat(), **counts} for d, counts in days.items()]


async def overview(
    conn: AsyncConnection, *, period: Period, source: Source, now: datetime
) -> dict[str, Any]:
    """Counts for the Today page. Rates are left to the page, which shows each with its n."""
    start = _period_start(period, now)
    rows, facts = await _window(conn, source=source, start=start)
    results = Counter(
        r.outcome.value if r.outcome else call_status(None, r.created_at, now) for r in rows
    )
    headline = _headline(rows, facts)
    previous = None
    if (window := _previous_window(period, start, now)) is not None:
        previous = _headline(*await _window(conn, source=source, start=window[0], end=window[1]))
    cond = _source_filter(source)
    live = await conn.scalar(
        select(func.count())
        .select_from(Conversation)
        .where(
            Conversation.outcome.is_(None),
            Conversation.created_at > now - LIVE_WINDOW,
            *([cond] if cond is not None else []),
        )
    )
    return {
        "period": period,
        "source": source,
        "labels": dict(Counter(r.label.value for r in rows)),
        **headline,
        "previous": previous,
        "in_progress": int(live or 0),
        "funnel": [
            {"step": "Calls made", "count": len(rows)},
            {"step": "Right person confirmed", "count": sum(f.verified for f in facts.values())},
            {"step": "Payment option offered", "count": sum(f.offers > 0 for f in facts.values())},
            {"step": "Agreed to pay", "count": headline["agreed_to_pay"]},
        ],
        "results": [
            {
                "result": key,
                "words": RESULT_WORDS.get(key) or STATUS_WORDS.get(key, key),
                "count": n,
            }
            for key, n in results.most_common()
        ],
        "per_day": _per_day(rows, facts, start, now),
        "due_soon": await _due_soon(conn, source=source, now=now),
    }


async def _due_soon(conn: AsyncConnection, *, source: Source, now: datetime) -> dict[str, Any]:
    """Payments due in the next DUE_SOON_DAYS days on promises that are still active."""
    today = now.astimezone(BUSINESS_TZ).date()
    until = today + timedelta(days=DUE_SOON_DAYS)
    stmt = (
        select(Event.input["schedule"])
        .select_from(Promise)
        .join(Event, Event.id == Promise.terms_event_id)
        .join(Conversation, Conversation.id == Promise.conversation_id)
        .where(Promise.status == PromiseStatus.ACTIVE)
    )
    if (cond := _source_filter(source)) is not None:
        stmt = stmt.where(cond)
    due = [
        p
        for schedule in (await conn.scalars(stmt))
        for p in schedule or []
        if today <= _d(p["due"]) <= until
    ]
    return {
        "days": DUE_SOON_DAYS,
        "count": len(due),
        "amount_jpy": sum(int(p["amount_jpy"]) for p in due),
    }


# --- Callbacks the customer asked for: dialed only once the owner approves (V5-3) -----------

CALLBACK_REASON_WORDS = {
    "payday": "waiting for payday",
    "needs_time": "needs some time",
    "busy_now": "busy right now",
    "other": "other",
}


class CallbackDecisionError(ValueError):
    """The callback cannot be approved or rejected now (wrong status, or its day has passed)."""


async def callbacks(
    conn: AsyncConnection, *, status: CallbackStatus | None, now: datetime
) -> list[dict[str, Any]]:
    stmt = (
        select(
            Callback.id,
            Callback.due_date,
            Callback.reason,
            Callback.customer_words,
            Callback.status,
            Callback.conversation_id,
            Callback.done_conversation_id,
            Callback.created_at,
            Debtor.full_name,
            Debtor.external_ref,
            Debtor.timezone,
        )
        .join(Debtor, Debtor.id == Callback.debtor_id)
        .order_by(Callback.due_date, Callback.id)
    )
    if status is not None:
        stmt = stmt.where(Callback.status == status)
    return [
        {
            "id": r.id,
            "customer": r.full_name,
            "customer_ref": r.external_ref,
            "due_date": r.due_date.isoformat(),
            "due_words": spoken_date(r.due_date),
            "overdue": r.due_date < now.astimezone(ZoneInfo(r.timezone)).date(),
            "reason": r.reason,
            "reason_words": CALLBACK_REASON_WORDS.get(r.reason, r.reason),
            "customer_words": r.customer_words,
            "status": r.status.value,
            "requested_in_call": r.conversation_id,
            "done_in_call": r.done_conversation_id,
            "requested_at": r.created_at.isoformat(),
        }
        for r in await conn.execute(stmt)
    ]


async def decide_callback(
    conn: AsyncConnection, callback_id: int, *, approve: bool, now: datetime
) -> dict[str, Any] | None:
    """The owner approves or rejects a requested callback. Each decision is an audit event.
    Approving a callback whose day has passed is refused: reject it instead."""
    row = (
        await conn.execute(
            select(
                Callback.status,
                Callback.due_date,
                Callback.debtor_id,
                Callback.conversation_id,
                Debtor.timezone,
            )
            .join(Debtor, Debtor.id == Callback.debtor_id)
            .where(Callback.id == callback_id)
            .with_for_update(of=Callback)
        )
    ).one_or_none()
    if row is None:
        return None
    if row.status is not CallbackStatus.REQUESTED:
        raise CallbackDecisionError(f"this callback is already {row.status.value}")
    if approve and row.due_date < now.astimezone(ZoneInfo(row.timezone)).date():
        raise CallbackDecisionError("the requested day has passed: reject it instead")
    status = CallbackStatus.APPROVED if approve else CallbackStatus.REJECTED
    await conn.execute(
        update(Callback).where(Callback.id == callback_id).values(status=status, decided_at=now)
    )
    await record_tool_call(
        conn,
        conversation_id=row.conversation_id,
        debtor_id=row.debtor_id,
        tool="admin_approve_callback" if approve else "admin_reject_callback",
        at=now,
        agent_version="admin",
        detail={"callback_id": callback_id, "due_date": row.due_date.isoformat()},
    )
    return {"id": callback_id, "status": status.value}
