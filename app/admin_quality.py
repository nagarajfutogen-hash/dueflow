"""Admin dashboard D-3 (PLANNING §16): Quality, for the owner. Technical words allowed here.

Everything is counted from stored rows for the calls in the chosen period, source and agent
version: the output guard (blocks, rewrites, fallbacks), every policy rule (allowed vs
blocked), the v5 payment ladder and callbacks. Counts only; the page shows each rate with n.
"""

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncConnection

from app.admin import Period, Source, _period_start, _source_filter
from app.agent.scripts import (
    SAFE_FALLBACK_POST,
    SAFE_FALLBACK_POST_V4,
    SAFE_FALLBACK_POST_V5,
    SAFE_FALLBACK_PRE,
)
from app.db.models import (
    Callback,
    Conversation,
    Decision,
    Event,
    EventType,
    Promise,
    ResultLabel,
    Speaker,
    Turn,
)

FALLBACKS = (SAFE_FALLBACK_POST, SAFE_FALLBACK_POST_V4, SAFE_FALLBACK_POST_V5, SAFE_FALLBACK_PRE)


def _calls(
    source: Source, period: Period, version: str | None, now: datetime
) -> Select[int, str, ResultLabel]:
    stmt = select(Conversation.id, Conversation.agent_version, Conversation.label)
    if (cond := _source_filter(source)) is not None:
        stmt = stmt.where(cond)
    if (start := _period_start(period, now)) is not None:
        stmt = stmt.where(Conversation.created_at >= start)
    if version:
        stmt = stmt.where(Conversation.agent_version == version)
    return stmt


async def quality(
    conn: AsyncConnection,
    *,
    period: Period,
    source: Source,
    version: str | None,
    now: datetime,
) -> dict[str, Any]:
    calls = list(await conn.execute(_calls(source, period, version, now)))
    ids = [c.id for c in calls]
    version_of = {c.id: c.agent_version for c in calls}
    labels = Counter(c.label.value for c in calls)
    versions = sorted({c.agent_version for c in calls})
    if not ids:
        return {
            "calls": 0,
            "labels": {},
            "versions": [],
            "guard": [],
            "block_reasons": [],
            "rules": [],
            "ladder": {"offers": [], "handoffs": [], "agreed": []},
            "callbacks": [],
        }

    # --- Output guard: blocks, what followed each block, and fallbacks said -------------------
    checks = await conn.execute(
        select(Event.conversation_id, Event.decision, Event.input["violation"].astext)
        .where(Event.conversation_id.in_(ids), Event.rule == "output_check")
        .order_by(Event.id)
    )
    per_call: dict[int, list[bool]] = defaultdict(list)
    reasons: Counter[str] = Counter()
    for cid, decision, violation in checks:
        blocked = decision == Decision.BLOCKED
        per_call[cid].append(blocked)
        if blocked:
            reasons[violation or "unknown"] += 1
    agent_turns = await conn.execute(
        select(Turn.conversation_id, Turn.text).where(
            Turn.conversation_id.in_(ids), Turn.speaker == Speaker.AGENT
        )
    )
    fallbacks: Counter[str] = Counter()
    for cid, text in agent_turns:
        if any(f in text for f in FALLBACKS):
            fallbacks[version_of[cid]] += 1
    guard: dict[str, Counter[str]] = defaultdict(Counter)
    for cid, seq in per_call.items():
        g = guard[version_of[cid]]
        g["checks"] += len(seq)
        g["blocked"] += sum(seq)
        for a, b in zip(seq, seq[1:], strict=False):
            if a:
                g["then_passed" if not b else "then_blocked"] += 1
    calls_by_version = Counter(version_of.values())

    # --- Every policy rule: allowed vs blocked --------------------------------------------------
    rules: dict[str, Counter[str]] = defaultdict(Counter)
    for rule, decision in await conn.execute(
        select(Event.rule, Event.decision).where(
            Event.conversation_id.in_(ids), Event.type == EventType.POLICY_DECISION
        )
    ):
        rules[rule or "unknown"][decision.value if decision else "none"] += 1

    # --- v5 payment ladder ------------------------------------------------------------------
    offers: Counter[str] = Counter()
    handoffs: Counter[str] = Counter()
    for decision, rung, codes in await conn.execute(
        select(Event.decision, Event.input["offer"]["rung"].astext, Event.input["codes"]).where(
            Event.conversation_id.in_(ids), Event.rule == "negotiation_ladder"
        )
    ):
        if decision == Decision.ALLOWED and rung:
            offers[rung] += 1
        elif codes:
            handoffs[str(codes[0])] += 1
    agreed: Counter[str] = Counter()
    for (rung,) in await conn.execute(
        select(Event.input["ladder_rung"].astext)
        .select_from(Promise)
        .join(Event, Event.id == Promise.terms_event_id)
        .where(Promise.conversation_id.in_(ids))
    ):
        agreed[rung or "not_from_the_ladder"] += 1

    callbacks = Counter(
        s.value
        for s in await conn.scalars(
            select(Callback.status).where(Callback.conversation_id.in_(ids))
        )
    )

    return {
        "calls": len(ids),
        "labels": dict(labels),
        "versions": versions,
        "guard": [
            {
                "version": v,
                "calls": calls_by_version[v],
                "checks": guard[v]["checks"],
                "blocked": guard[v]["blocked"],
                "then_passed": guard[v]["then_passed"],
                "then_blocked": guard[v]["then_blocked"],
                "fallbacks": fallbacks[v],
            }
            for v in versions
        ],
        "block_reasons": [{"reason": r, "count": n} for r, n in reasons.most_common()],
        "rules": [
            {"rule": r, "allowed": c["allowed"], "blocked": c["blocked"]}
            for r, c in sorted(rules.items(), key=lambda kv: -kv[1]["blocked"])
        ],
        "ladder": {
            "offers": [{"rung": r, "count": n} for r, n in offers.most_common()],
            "handoffs": [{"code": r, "count": n} for r, n in handoffs.most_common()],
            "agreed": [{"rung": r, "count": n} for r, n in agreed.most_common()],
        },
        "callbacks": [{"status": s, "count": n} for s, n in callbacks.most_common()],
    }
