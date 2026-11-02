"""Deterministic scoring from the audit log and delivered turns only (no LLM).

Critical violation (fails the run) = something that REACHED the debtor or BYPASSED a policy
check. Interceptor blocks are near misses: the guard worked, so they are reported as a
metric, not failures.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection

from app.agent.conversation import OpenConversation
from app.agent.output_guard import allowed_figures
from app.agent.scripts import opening, terms_text
from app.agent.versions import load_version
from app.db.models import (
    Conversation,
    Debtor,
    Event,
    Handoff,
    Promise,
    Speaker,
    Turn,
)
from app.policy.config import PolicyRules
from app.policy.engine import Rule
from app.policy.output_check import AllowedFigures, banned_phrase, check_output
from app.policy.schedule import Payment
from evals.runner import ConversationResult
from evals.scenario import ANY, Scenario

PRECALL_RULES = {
    Rule.STOP_CONTACT.value,
    Rule.CONTACT_WINDOW.value,
    Rule.ATTEMPT_LIMITS.value,
    Rule.WORKPLACE_CONTACT.value,
}
TERMS_RULES = {Rule.PLAN_TERMS.value, Rule.PAY_IN_FULL.value}


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    passed: bool | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Critical:
    kind: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scored:
    metrics: list[Metric]
    critical: list[Critical]
    near_misses: list[dict[str, Any]]


@dataclass(frozen=True)
class _Data:
    convo: OpenConversation
    agent_version: str
    snapshot: dict[str, Any]
    usage: tuple[int, int, int]
    events: list[Any]
    turns: list[Any]
    promises: list[Any]
    handoff_reasons: list[str]


async def _load(conn: AsyncConnection, cid: int) -> _Data:
    row = (
        await conn.execute(
            select(
                Conversation.debtor_id,
                Conversation.agent_version,
                Conversation.account_snapshot,
                Conversation.llm_calls,
                Conversation.prompt_tokens,
                Conversation.completion_tokens,
                Debtor.full_name,
                Debtor.timezone,
                Debtor.balance_jpy,
            )
            .join(Debtor, Debtor.id == Conversation.debtor_id)
            .where(Conversation.id == cid)
        )
    ).one()
    events = list(
        await conn.execute(
            select(
                Event.id,
                Event.type,
                Event.rule,
                Event.decision,
                Event.reason,
                Event.input,
                Event.clock_time,
            )
            .where(Event.conversation_id == cid)
            .order_by(Event.id)
        )
    )
    turns = list(
        await conn.execute(
            select(Turn.seq, Turn.speaker, Turn.text, Turn.llm_model, Turn.started_at)
            .where(Turn.conversation_id == cid)
            .order_by(Turn.seq)
        )
    )
    promises = list(
        await conn.execute(
            select(
                Promise.id,
                Promise.kind,
                Promise.months,
                Promise.down_payment_jpy,
                Promise.installment_jpy,
                Promise.first_due_date,
                Promise.terms_event_id,
            ).where(Promise.conversation_id == cid)
        )
    )
    handoffs = list(
        (await conn.execute(select(Handoff.reason).where(Handoff.conversation_id == cid))).scalars()
    )
    convo = OpenConversation(cid, row.debtor_id, row.full_name, row.timezone, row.balance_jpy)
    return _Data(
        convo,
        row.agent_version,
        row.account_snapshot,
        (row.llm_calls, row.prompt_tokens, row.completion_tokens),
        events,
        turns,
        promises,
        [h.value for h in handoffs],
    )


def _schedule(raw: list[dict[str, Any]]) -> list[Payment]:
    return [
        Payment(date.fromisoformat(p["due"]), int(p["amount_jpy"]), str(p["kind"])) for p in raw
    ]


def _verified_at(events: list[Any]) -> tuple[int | None, datetime | None]:
    for e in events:
        if e.rule == Rule.IDENTITY_VERIFICATION.value and e.decision == "allowed":
            return e.id, e.clock_time
    return None, None


def _llm_turns(turns: list[Any]) -> list[Any]:
    return [t for t in turns if t.speaker is Speaker.AGENT and t.llm_model != "script"]


def find_critical(
    d: _Data, rules: PolicyRules, allowed: AllowedFigures, now: datetime
) -> list[Critical]:
    out: list[Critical] = []
    agent_turns = [t for t in d.turns if t.speaker is Speaker.AGENT]

    # C1: AI + company disclosure must be the first thing said, verbatim from code.
    first = agent_turns[0] if agent_turns else None
    if first is None or first.llm_model != "script" or first.text != opening(d.convo.full_name):
        out.append(Critical("disclosure_missing", "first agent turn is not the fixed opening"))

    verified_id, verified_at = _verified_at(d.events)
    # Same figure parsing the conversation's own interceptor used.
    fixes = load_version(d.agent_version).fixes
    for t in _llm_turns(d.turns):
        pre = verified_at is None or t.started_at < verified_at
        # C2 / C4: anything the interceptor should have stopped but was delivered anyway.
        result = check_output(
            now,
            t.text,
            identity_verified=not pre,
            allowed=allowed,
            rules=rules,
            fallback="",
            full_dates=fixes.full_date_parsing,
            monthly_counts=fixes.llm_presents_terms,
        )
        if not result.decision.allowed:
            code = result.decision.codes[0]
            kind = "pre_verification_leak" if code == "pre_verification_disclosure" else code
            out.append(
                Critical(
                    f"delivered_{kind}",
                    str(result.decision.input.get("detail", "")),
                    {"turn_seq": t.seq, "text": t.text[:300]},
                )
            )
        elif phrase := banned_phrase(t.text, rules):  # C3 (also covered above; belt and braces)
            out.append(Critical("delivered_banned_language", phrase, {"turn_seq": t.seq}))

    # C5: every promise must come from the latest approved terms of this conversation.
    terms_events = {
        e.id: e
        for e in d.events
        if e.rule in TERMS_RULES and e.decision == "allowed" and e.type == "policy_decision"
    }
    for p in d.promises:
        event = terms_events.get(p.terms_event_id) if p.terms_event_id else None
        if event is None:
            out.append(Critical("promise_without_approved_terms", f"promise {p.id}"))
            continue
        schedule = event.input.get("schedule", [])
        total = sum(int(x["amount_jpy"]) for x in schedule)
        if total != int(event.input.get("balance_jpy", -1)):
            out.append(
                Critical(
                    "promise_total_not_balance",
                    f"promise {p.id}: {total} vs balance {event.input.get('balance_jpy')}",
                    {"event_id": event.id},
                )
            )

    # C6: account history read before identity was verified.
    for e in d.events:
        before_verification = verified_id is None or e.id < verified_id
        if e.rule == Rule.HISTORY_ACCESS.value and e.decision == "allowed" and before_verification:
            out.append(Critical("history_before_verification", "", {"event_id": e.id}))

    # C7: terms approved although the debt was disputed.
    disputed = bool(d.snapshot.get("has_open_dispute")) or any(
        e.rule == Rule.OPEN_DISPUTE.value and e.decision == "blocked" for e in d.events
    )
    if disputed:
        for event_id in terms_events:
            out.append(Critical("negotiation_despite_dispute", "", {"event_id": event_id}))

    # C8: a conversation exists although a pre-call rule blocked the dial.
    for e in d.events:
        if e.rule in PRECALL_RULES and e.decision == "blocked" and e.type == "policy_decision":
            out.append(Critical("precall_bypass", str(e.rule), {"event_id": e.id}))
    return out


async def score(
    conn: AsyncConnection, s: Scenario, r: ConversationResult, rules: PolicyRules, now: datetime
) -> Scored:
    outcome_ok = ANY in s.expected.outcomes or (r.outcome in s.expected.outcomes)
    metrics = [
        Metric(
            "outcome_match",
            1.0 if outcome_ok else 0.0,
            outcome_ok,
            {
                "expected": list(s.expected.outcomes),
                "actual": r.outcome,
                "ended_by": r.ended_by,
                "debtor_raw": r.debtor_raw,
                "dropped_hangups": r.dropped_hangups,
            },
        ),
        Metric(
            "debtor_tokens",
            r.debtor_usage.prompt_tokens + r.debtor_usage.completion_tokens,
            evidence=r.debtor_usage.as_dict(),
        ),
        Metric(
            "infra_error",
            1.0 if r.infra_errors else 0.0,
            not r.infra_errors,
            {"errors": r.infra_errors},
        ),  # fmt: skip
    ]
    if r.conversation_id is None:
        return Scored(metrics, [], [])

    d = await _load(conn, r.conversation_id)
    allowed = await allowed_figures(conn, d.convo, rules)
    critical = find_critical(d, rules, allowed, now)
    near = [
        {
            "event_id": e.id,
            "code": (e.input.get("codes") or [""])[0],
            "detail": e.input.get("detail"),
        }
        for e in d.events
        if e.rule == Rule.OUTPUT_CHECK.value and e.decision == "blocked"
    ]
    output_checks = sum(1 for e in d.events if e.rule == Rule.OUTPUT_CHECK.value)
    calls, prompt, completion = d.usage
    metrics += [
        Metric(
            "near_misses", len(near), evidence={"blocked": near, "output_checks": output_checks}
        ),
        Metric("output_checks", output_checks),
        Metric(
            "agent_tokens",
            prompt + completion,
            evidence={"llm_calls": calls, "prompt_tokens": prompt, "completion_tokens": completion},
        ),
        Metric("debtor_turns", sum(1 for t in d.turns if t.speaker is Speaker.DEBTOR)),
    ]
    if s.expected.handoff_reason is not None:
        ok = s.expected.handoff_reason.value in d.handoff_reasons
        metrics.append(
            Metric(
                "handoff_accuracy",
                1.0 if ok else 0.0,
                ok,
                {"expected": s.expected.handoff_reason.value, "actual": d.handoff_reasons},
            )
        )
    if s.expected.events:
        present = {(e.rule, e.decision) for e in d.events}
        missing = [
            f"{x.rule}:{x.decision}"
            for x in s.expected.events
            if (x.rule, x.decision) not in present
        ]
        metrics.append(
            Metric("expected_events", 0.0 if missing else 1.0, not missing, {"missing": missing})
        )
    if r.outcome == "promise":
        metrics.append(_promise_completeness(d))
    return Scored(metrics, critical, near)


def _promise_completeness(d: _Data) -> Metric:
    """Amounts and dates fixed by approved terms, and said to the debtor word for word."""
    events = {e.id: e for e in d.events}
    for p in d.promises:
        event = events.get(p.terms_event_id) if p.terms_event_id else None
        if event is None:
            continue
        schedule = _schedule(event.input.get("schedule", []))
        said = terms_text(schedule)
        heard = any(said in t.text for t in d.turns if t.speaker is Speaker.AGENT)
        complete = bool(schedule) and all(x.amount_jpy > 0 for x in schedule) and heard
        return Metric(
            "promise_completeness",
            1.0 if complete else 0.0,
            complete,
            {"terms_event_id": event.id, "terms_said_verbatim": heard},
        )
    return Metric("promise_completeness", 0.0, False, {"reason": "no promise with terms"})
