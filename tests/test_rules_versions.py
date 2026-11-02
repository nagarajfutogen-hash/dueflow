"""rules-v6 -> rules-v7 changes nothing for agent v1/v2.

This is what makes `--allow-rules-change rules-v6:rules-v7` safe: v1 (champion, measured on
rules-v6) can be compared with v3 (rules-v7). Two proofs:
1. Every policy decision v1/v2 can reach is identical under both files (only the recorded
   rules_version label differs).
2. v1/v2 agent paths that v3 changed run with rules-v6 (no v7 lists at all) without touching
   them: the v7 rule functions raise on rules-v6, so any read would fail these tests.
"""

import asyncio
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pipecat.flows import FlowManager
from pipecat.pipeline.worker import PipelineWorker
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent import scripts
from app.agent.live import LiveChat
from app.agent.tools import make_handlers
from app.db.models import Event, Outcome
from app.policy import engine
from app.policy.config import PolicyRules, load_rules
from app.policy.engine import Decision, DialTarget
from app.policy.output_check import AllowedFigures, check_output
from tests.conftest import MakeDebtor
from tests.policy.conftest import jst
from tests.test_agent_versions import StubFlow, StubWorker
from tests.test_chat_offline import NAME, _session

V6 = load_rules(Path("tests/fixtures/rules-v6.yaml"))
V7 = load_rules(Path("tests/fixtures/rules-v7.yaml"))
V8 = load_rules()
NOW = jst(2026, 10, 24, 12)
V7_RULES = {engine.Rule.WRONG_NUMBER_CLAIM.value, engine.Rule.PAYMENT_CLAIM.value}


def test_fixture_is_rules_v6_and_current_is_rules_v7() -> None:
    assert (V6.version, V7.version, V8.version) == ("rules-v6", "rules-v7", "rules-v8")
    assert V6.third_party is None and V6.dispute is None
    # Every rules-v6 value is unchanged in rules-v7.
    shared = V7.model_dump(exclude={"version", "third_party", "dispute"})
    assert shared == V6.model_dump(exclude={"version", "third_party", "dispute"})


# --- 1. Policy decisions ---------------------------------------------------------------------


def _same(a: Decision, b: Decision) -> None:
    strip = {k: v for k, v in a.input.items() if k != "rules_version"}
    assert (a.rule, a.allowed, a.reason, a.codes) == (b.rule, b.allowed, b.reason, b.codes)
    assert strip == {k: v for k, v in b.input.items() if k != "rules_version"}


ALLOWED = AllowedFigures(
    amounts=frozenset({100_000, 10_000, 30_000}),
    counts=frozenset({3, 4, 5, 6}),
    percents=frozenset({10, 20}),
    dates=frozenset({date(2026, 10, 31), date(2027, 3, 1)}),
)
REPLIES = [
    "Your balance is ¥100,000.",
    "Pay ¥25,000 a month.",
    "We will take legal action.",
    "It's about your overdue loan.",
    "The first payment is due March 1, 2027.",
    "Can you pay by tomorrow?",
]


def _decisions(rules: PolicyRules) -> list[Decision]:
    out = [
        engine.check_contact_window(jst(2026, 10, 24, h), "Asia/Tokyo", rules) for h in (7, 8, 21)
    ]
    out.append(engine.check_attempt_limits(NOW, "Asia/Tokyo", [jst(2026, 10, 24, 9)], rules))
    out.append(engine.check_workplace_contact(NOW, DialTarget.WORK, False, None))
    for months, down, broken in [(3, 10_000, False), (12, 0, False), (4, 10_000, True)]:
        out.append(
            engine.check_plan_terms(
                NOW,
                balance_jpy=100_000,
                months=months,
                down_payment_jpy=down,
                has_broken_promise=broken,
                rules=rules,
            )
        )
    for when in (date(2026, 10, 27), date(2026, 11, 23)):  # inside / outside the 14 days
        out.append(
            engine.check_pay_in_full(
                NOW,
                debtor_tz="Asia/Tokyo",
                balance_jpy=100_000,
                amount_jpy=100_000,
                payment_date=when,
                rules=rules,
            )
        )
    out += [
        engine.check_no_discount(NOW, balance_jpy=100_000, offered_total_jpy=x)
        for x in (0, 50_000, 100_000)
    ]
    for text, failed in [("Sora Test, 1990-01-01", 0), ("Sora Test", 0), ("x, 1991-02-02", 1)]:
        out.append(
            engine.check_identity(
                NOW,
                debtor_text=text,
                actual_name="Sora Test",
                actual_dob=date(1990, 1, 1),
                failed_attempts=failed,
                rules=rules,
            )
        )
    for full_dates in (False, True):
        for verified in (False, True):
            out += [
                check_output(
                    NOW,
                    t,
                    identity_verified=verified,
                    allowed=ALLOWED,
                    rules=rules,
                    fallback="F.",
                    full_dates=full_dates,
                ).decision
                for t in REPLIES
            ]
    return out


def test_every_v1_v2_policy_decision_is_identical_under_v6_and_v7() -> None:
    v6, v7 = _decisions(V6), _decisions(V7)
    assert len(v6) == len(v7) == 40
    for a, b in zip(v6, v7, strict=True):
        _same(a, b)


# --- 2. Agent paths v3 changed, run by v1/v2 on rules-v6 -------------------------------------


async def _live_on(
    rules: PolicyRules,
    client: httpx.AsyncClient,
    make_debtor: MakeDebtor,
    engine_: AsyncEngine,
    version: str,
) -> tuple[LiveChat, StubWorker, Any]:
    session, rec, _ = await _session(client, make_debtor, engine_, version)
    session = replace(session, rules=rules)
    log: list[str] = []
    worker = StubWorker(log)
    live = LiveChat(
        session,
        cast(PipelineWorker, worker),
        cast(FlowManager, StubFlow(log)),
        make_handlers(session),
        cast("asyncio.Task[Any]", None),
    )
    return live, worker, rec


async def _outcome_of(
    rules: PolicyRules,
    version: str,
    client: httpx.AsyncClient,
    make_debtor: MakeDebtor,
    engine_: AsyncEngine,
) -> dict[str, Any]:
    live, worker, rec = await _live_on(rules, client, make_debtor, engine_, version)
    s, h = live.session, live.handlers
    # Wrong number on "What do you want?" (v3 would refuse it).
    await live.debtor_says("What do you want?")
    wrong = await h["report_third_party"]({"kind": "wrong_number"}, live.flow)

    live2, worker2, _ = await _live_on(rules, client, make_debtor, engine_, version)
    # Verify (v1: by the LLM's tool call; v2: by code), then claim it's paid, then offer ¥0.
    await live2.debtor_says(f"{NAME}, 1993-08-08")
    if live2.session.node == "identify":
        await live2.handlers["verify_identity"]({}, live2.flow)
    await live2.debtor_says("I already paid this in full.")
    discount, _ = await live2.handlers["check_discount_request"]({"offered_total_jpy": 0}, None)  # type: ignore[arg-type]
    async with engine_.connect() as conn:
        rules_hit = sorted(  # no ORDER BY: compare as a sorted list, not in insertion order
            str(rule)
            for rule in (
                await conn.execute(
                    select(Event.rule).where(
                        Event.conversation_id.in_(
                            [s.conversation_id, live2.session.conversation_id]
                        ),
                        Event.rule.in_(V7_RULES),
                    )
                )
            ).scalars()
        )
    return {
        "wrong_number": (wrong[0]["status"], s.outcome, rec.said()[-1:]),
        "after_paid_claim": (live2.session.outcome, worker2.appended()[-1]),
        "discount": discount,
        "v7_rule_events": rules_hit,
    }


@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_v1_v2_agent_paths_never_read_rules_v7(
    api_client: httpx.AsyncClient,
    make_debtor: MakeDebtor,
    app_engine: AsyncEngine,
    version: str,
) -> None:
    on_v6 = await _outcome_of(V6, version, api_client, make_debtor, app_engine)
    on_v7 = await _outcome_of(V7, version, api_client, make_debtor, app_engine)
    assert on_v6 == on_v7
    assert on_v6["wrong_number"] == (
        "ended",
        Outcome.END_WRONG_PERSON.value,
        [scripts.WRONG_NUMBER_END],
    )
    assert on_v6["after_paid_claim"] == (None, ("user", True))  # the LLM answers, no handoff
    assert on_v6["discount"]["reason"].startswith("offered ¥0 is less than balance")
    assert on_v6["v7_rule_events"] == []


# --- 3. rules-v7 -> rules-v8 (2026-10-31): v5-only blocks, nothing else ----------------------


def test_rules_v8_only_adds_the_v5_blocks() -> None:
    assert V7.ladder is None and V7.callback is None and V7.claims is None
    assert V8.ladder is not None and V8.callback is not None and V8.claims is not None
    v5_only = {"version", "ladder", "callback", "claims"}
    assert V8.model_dump(exclude=v5_only) == V7.model_dump(exclude=v5_only)


def test_every_policy_decision_is_identical_under_v7_and_v8() -> None:
    v7, v8 = _decisions(V7), _decisions(V8)
    assert len(v7) == len(v8) == 40
    for a, b in zip(v7, v8, strict=True):
        _same(a, b)


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
async def test_v1_to_v4_agent_paths_are_identical_under_v7_and_v8(
    api_client: httpx.AsyncClient,
    make_debtor: MakeDebtor,
    app_engine: AsyncEngine,
    version: str,
) -> None:
    on_v7 = await _outcome_of(V7, version, api_client, make_debtor, app_engine)
    on_v8 = await _outcome_of(V8, version, api_client, make_debtor, app_engine)
    assert on_v7 == on_v8
