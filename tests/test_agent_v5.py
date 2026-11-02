"""Agent v5 (rules-v8): the engine picks every offer down its payment ladder; the LLM talks.

Every v5 switch is tested on v5 (new behavior) and on v4 (unchanged), allowed and blocked.
Account used: ¥100,000 owed, 10% minimum down (¥10,000), 3-6 months allowed.
"""

from dataclasses import replace
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent import scripts
from app.agent.nodes import negotiate_node
from app.agent.session import ChatSession
from app.agent.tools import Handler, make_handlers
from app.agent.versions import load_version
from app.db.models import Event, Handoff, HandoffReason, Outcome, Promise, Turn
from app.policy.config import load_rules
from tests.conftest import MakeDebtor
from tests.test_agent_v4 import _said
from tests.test_chat_offline import FM, _session, _verify

RULES = load_rules()
V4, V5 = load_version("v4"), load_version("v5")
V5_FIXES = (
    "payment_ladder",
    "one_option_at_a_time",
    "no_interest_claims",
    "callbacks",
    "echo_month_year",
    "wrap_up_question",
)


async def _ready(
    client: httpx.AsyncClient, make_debtor: MakeDebtor, engine: AsyncEngine, **debtor: Any
) -> tuple[ChatSession, dict[str, Handler]]:
    session, _, h = await _session(client, make_debtor, engine, "v5", **debtor)
    await _verify(session, h)
    assert session.node == "negotiate"
    return session, h


async def _offer(h: dict[str, Handler], reason: str, **amounts: int) -> dict[str, Any]:
    out, _ = await h["next_offer"]({"reason": reason, **amounts}, FM)
    return out


async def _handoff(engine: AsyncEngine, conversation_id: int) -> Any:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                select(Handoff.reason, Handoff.summary).where(
                    Handoff.conversation_id == conversation_id
                )
            )
        ).one()


async def _ladder_codes(engine: AsyncEngine, conversation_id: int) -> list[tuple[str, Any]]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            select(Event.decision, Event.input)
            .where(Event.conversation_id == conversation_id, Event.rule == "negotiation_ladder")
            .order_by(Event.id)
        )
        return [(r.decision.value, r.input.get("codes")) for r in rows]


# --- Version file and prompt ----------------------------------------------------------------


def test_v5_is_v4_plus_the_ladder_on_rules_v8() -> None:
    v4_on = {k for k, v in V4.fixes.model_dump().items() if v}
    v5_on = {k for k, v in V5.fixes.model_dump().items() if v}
    assert v5_on == v4_on | set(V5_FIXES)
    assert (V5.prompt_version, V5.rules_version, V5.temperature) == (
        "prompts-v5.12",
        RULES.version,
        V4.temperature,
    )
    assert not any(getattr(V4.fixes, f) for f in V5_FIXES)


def test_v5_prompt_lets_the_system_pick_options_v4_unchanged() -> None:
    v5 = negotiate_node({"balance_jpy": 100_000}, RULES, V5)
    v4 = negotiate_node({"balance_jpy": 100_000}, RULES, V4)
    v5_tools = {t.name: t for t in v5.tools}
    assert "next_offer" in v5_tools and "next_offer" not in {t.name for t in v4.tools}
    # v5: the system hands off for "can't afford"; the LLM keeps dispute and unresolved only.
    assert v5_tools["request_handoff"].properties["reason"]["enum"] == ["dispute", "unresolved"]
    assert "chosen by the system, never by you" in v5.task
    assert "one payment option at a time" in v5.role
    assert "a colleague can confirm" in v5.role
    assert "AVOID:" in v5.task  # example replies per approach, including what not to say
    # v4 prompt keeps its own lines and gets none of v5's.
    assert "what's been going on" in v4.task and "next_offer" not in v4.task
    assert "one payment option at a time" not in v4.role


# --- The ladder, one option at a time ---------------------------------------------------------


async def test_v5_ladder_moves_down_then_hands_off_at_the_counter_limit(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, h = await _ready(api_client, make_debtor, app_engine)
    full = await _offer(h, "first_offer")
    assert full["option"] == "pay_in_full"
    assert full["terms"]["full_payment"] == "¥100,000"
    standard = await _offer(h, "cannot_pay_in_full")
    assert standard["option"] == "standard_split"
    assert (standard["terms"]["monthly_payments"], standard["terms"]["monthly_amount"]) == (
        3,
        "¥30,000",
    )
    extended = await _offer(h, "monthly_too_high")
    assert extended["option"] == "extended_split"
    assert (extended["terms"]["monthly_payments"], extended["terms"]["monthly_amount"]) == (
        6,
        "¥15,000",
    )
    end = await _offer(h, "monthly_too_high")
    assert end == {"status": "ended", "handoff": "unresolved"}  # refused all: no amount said
    assert session.outcome == Outcome.HANDOFF.value
    reason, summary = await _handoff(app_engine, session.conversation_id)
    assert reason is HandoffReason.UNRESOLVED  # PLANNING §6 rule 3
    assert "did not accept the allowed options" in summary
    codes = await _ladder_codes(app_engine, session.conversation_id)
    assert codes == [("allowed", [])] * 3 + [("blocked", ["counter_limit"])]


async def test_v5_engine_offer_can_be_promised(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, h = await _ready(api_client, make_debtor, app_engine)
    await _offer(h, "first_offer")
    standard = await _offer(h, "cannot_pay_in_full")
    await session.guard_and_send(_said(standard["terms"]), None, None)
    done, _ = await h["record_promise"]({"terms_id": standard["terms_id"]}, FM)
    assert done["recorded"] is True
    async with app_engine.connect() as conn:
        months = await conn.scalar(
            select(Promise.months).where(Promise.conversation_id == session.conversation_id)
        )
    assert months == 3


async def test_v5_stated_amount_below_every_plan_hands_off_below_the_floor(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, h = await _ready(api_client, make_debtor, app_engine)
    await _offer(h, "first_offer")
    end = await _offer(h, "monthly_too_high", affordable_monthly_jpy=5_000)
    assert end["handoff"] == "below_account_floor"
    reason, summary = await _handoff(app_engine, session.conversation_id)
    assert reason is HandoffReason.BELOW_ACCOUNT_FLOOR
    assert "cannot pay what the lowest allowed plan needs" in summary
    assert (await _ladder_codes(app_engine, session.conversation_id))[-1] == (
        "blocked",
        ["below_account_floor"],
    )


async def test_v5_stated_amount_that_fits_skips_to_that_plan(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    _, h = await _ready(api_client, make_debtor, app_engine)
    await _offer(h, "first_offer")
    out = await _offer(h, "monthly_too_high", affordable_monthly_jpy=20_000)
    assert out["option"] == "extended_split" and out["terms"]["monthly_amount"] == "¥15,000"


async def test_v5_customer_limits_from_the_file_shape_the_offers(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    _, h = await _ready(api_client, make_debtor, app_engine, max_months=4)
    await _offer(h, "first_offer")
    await _offer(h, "cannot_pay_in_full")
    extended = await _offer(h, "monthly_too_high")
    assert (extended["terms"]["monthly_payments"], extended["terms"]["monthly_amount"]) == (
        4,
        "¥22,500",
    )


async def test_v5_already_offered_comes_from_the_audit_trail_not_memory(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    """A reconnect (new session object, empty memory) still never re-offers a rung."""
    session, h = await _ready(api_client, make_debtor, app_engine)
    await _offer(h, "first_offer")
    fresh = replace(session, offered_plans={}, offered_schedule=(), said_since_terms=[])
    out = await _offer(make_handlers(fresh), "cannot_pay_in_full")
    assert out["option"] == "standard_split"


async def test_v5_open_dispute_still_blocks_offers(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, h = await _session(
        api_client, make_debtor, app_engine, "v5", open_dispute="says it was repaid"
    )
    await _verify(session, h)
    if session.outcome is None:  # the open dispute may already have ended the call
        await _offer(h, "first_offer")
    assert session.outcome == Outcome.HANDOFF.value
    reason, _ = await _handoff(app_engine, session.conversation_id)
    assert reason is HandoffReason.DISPUTE


# --- The customer's own plan -------------------------------------------------------------------


async def test_v5_customer_plan_inside_the_rules_is_accepted(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    _, h = await _ready(api_client, make_debtor, app_engine)
    out, _ = await h["propose_terms"]({"months": 4, "down_payment_jpy": 10_000}, FM)
    assert out["approved"] is True and out["option"] == "customer_plan"
    assert out["terms"]["monthly_payments"] == 4
    assert "their_plan_possible" not in out


async def test_v5_customer_plan_not_allowed_gets_the_engines_option_v4_a_refusal(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    _, h = await _ready(api_client, make_debtor, app_engine)
    out, _ = await h["propose_terms"]({"months": 12, "down_payment_jpy": 10_000}, FM)
    assert out["approved"] is True and out["option"] == "extended_split"
    assert out["their_plan_possible"] is False
    assert out["instruction"].startswith("Their own plan is not possible")
    assert out["terms"]["monthly_payments"] == 6

    session4, _, h4 = await _session(api_client, make_debtor, app_engine, "v4")
    await _verify(session4, h4)
    v4, _ = await h4["propose_terms"]({"months": 12, "down_payment_jpy": 10_000}, FM)
    assert v4["approved"] is False and "option" not in v4
    assert v4["refusal_codes"] == ["months_out_of_range"]


# --- No claims about interest or fees ----------------------------------------------------------


async def test_v5_never_claims_no_interest_v4_unchanged(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    claim = "There is no interest added to your balance."
    honest = "I don't have the interest details; a colleague can confirm that for you."
    for version in ("v5", "v4"):
        session, _, h = await _session(api_client, make_debtor, app_engine, version)
        await _verify(session, h)
        assert (await session.guard_and_send(claim, None, None) == claim) is (version == "v4")
        assert await session.guard_and_send(honest, None, None) == honest
    async with app_engine.connect() as conn:
        blocked = await conn.scalar(
            select(Event.input["violation"].astext)
            .where(
                Event.rule == "output_check",
                Event.input["offending_sentence"].astext == claim,
                Event.agent_version == "v5",
            )
            .order_by(Event.id.desc())
            .limit(1)
        )
    assert blocked == "unverified_claim"


# --- V5-5 fixes from the V5-4 live test ------------------------------------------------------


async def test_v5_pay_in_full_date_comes_from_the_customers_words(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, h = await _ready(api_client, make_debtor, app_engine)
    await session.record_debtor("I can pay everything on Friday.")
    out, _ = await h["propose_pay_in_full"]({"day_phrase": "Friday"}, FM)
    assert out["approved"] is True
    assert out["terms"]["full_payment"] == "¥100,000"
    async with app_engine.connect() as conn:
        due = await conn.scalar(
            select(Event.input["payment_date"].astext)
            .where(Event.conversation_id == session.conversation_id, Event.rule == "pay_in_full")
            .order_by(Event.id.desc())
            .limit(1)
        )
    assert due == "2026-10-30"  # NOW is Saturday 24 October: the next Friday


async def test_v5_pay_in_full_day_never_said_is_refused_and_audited(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, h = await _ready(api_client, make_debtor, app_engine)
    await session.record_debtor("I can pay everything soon.")
    out, _ = await h["propose_pay_in_full"]({"day_phrase": "2026-10-27"}, FM)
    assert out["approved"] is False and out["refusal_codes"] == ["date_not_said"]
    assert "never suggest a day" in out["instruction"]
    async with app_engine.connect() as conn:
        codes = await conn.scalar(
            select(Event.input["codes"])
            .where(Event.conversation_id == session.conversation_id, Event.rule == "pay_in_full")
            .order_by(Event.id.desc())
            .limit(1)
        )
    assert codes == ["date_not_said"]


async def test_v5_unresolved_handoff_is_checked_once_then_goes_through(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, h = await _ready(api_client, make_debtor, app_engine)
    args = {"reason": "unresolved", "summary": "Customer could not agree."}
    first, _ = await h["request_handoff"](args, FM)
    assert first["handed_off"] is False and "next_offer" in first["instruction"]
    assert session.outcome is None
    second, _ = await h["request_handoff"](args, FM)
    assert second.get("handed_off") is not False
    assert session.outcome == Outcome.HANDOFF.value


async def test_v5_dispute_handoff_is_never_held_back(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, h = await _ready(api_client, make_debtor, app_engine)
    await h["request_handoff"]({"reason": "dispute", "summary": "Says it is not their loan."}, FM)
    assert session.outcome == Outcome.HANDOFF.value


async def test_v5_cannot_pay_in_full_first_gets_a_plan(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    _, h = await _ready(api_client, make_debtor, app_engine)
    out = await _offer(h, "cannot_pay_in_full")
    assert out["option"] == "standard_split"


def test_v5_prompt_asks_for_a_human_voice_v4_unchanged() -> None:
    """V5-5: the V5-4 transcripts read like a call-centre script."""
    v5 = negotiate_node({"balance_jpy": 100_000}, RULES, V5)
    v4 = negotiate_node({"balance_jpy": 100_000}, RULES, V4)
    for rule in (
        "never open with a thank-you",
        "like 'November 24th'",
        "never greet",
        "I'm sorry to hear that",
    ):
        assert rule in v5.role + v5.task
        assert rule not in v4.role + v4.task
    assert "Does that make sense so far?" not in v5.task  # the stock line it copied


async def test_v5_blocked_reply_ends_with_the_v5_fallback(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _ = await _ready(api_client, make_debtor, app_engine)
    said = await session.guard_and_send("You could pay ¥5,000 instead.", None, None)
    assert said == scripts.SAFE_FALLBACK_POST_V5


# --- V5-6: what still sounded scripted, or looped, in the V5-5 re-run -----------------------


async def test_v5_code_says_the_opening_and_the_llm_waits(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine, "v5")
    _, node = await _verify(session, h)
    assert rec.said()[-1] == "Your balance with us is ¥100,000. How are things going with it?"
    assert node["respond_immediately"] is False


async def test_v4_opening_is_unchanged(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine, "v4")
    _, node = await _verify(session, h)
    assert "How are things going with it?" not in rec.said()[-1]
    assert node["respond_immediately"] is True


async def test_v5_offer_dates_drop_this_year(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    _, h = await _ready(api_client, make_debtor, app_engine)
    full = await _offer(h, "first_offer")
    assert full["terms"]["full_payment_by"] == "November 7th"  # NOW is 24 October 2026


async def test_v5_stock_opener_is_dropped_but_never_the_only_sentence(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _ = await _ready(api_client, make_debtor, app_engine)
    said = await session.guard_and_send("Thanks for letting me know. What's been going on?", 1, 1)
    assert said == "What's been going on?"
    assert await session.guard_and_send("Thank you.", 1, 1) == "Thank you."


async def test_v5_second_amount_question_goes_back_to_the_llm(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, h = await _ready(api_client, make_debtor, app_engine)
    asked: list[str] = []

    async def retry(feedback: str) -> None:
        asked.append(feedback)

    session.request_retry = retry
    await _offer(h, "first_offer")
    ask = "What's the most you could pay each month?"
    assert await session.guard_and_send(ask, 1, 1) == ask  # the first ask is fine
    assert await session.guard_and_send(ask, 1, 1) == ""  # the second goes back
    assert len(asked) == 1 and "next_offer" in asked[0]
    standard = await _offer(h, "other")  # a new offer: one ask allowed again
    assert standard["option"] == "standard_split"
    assert await session.guard_and_send(ask, 1, 1) == ask


async def test_v5_blocked_twice_new_offer_is_said_by_code(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine, "v5")
    await _verify(session, h)
    await _offer(h, "first_offer")
    standard = await _offer(h, "other")
    said = await session.guard_and_send("So 8 months would be fine.", None, None)
    assert said.startswith("You could pay ¥10,000 by October 31st, then 3 monthly payments")
    assert said == rec.said()[-1]
    done, _ = await h["record_promise"]({"terms_id": standard["terms_id"]}, FM)
    assert done["recorded"] is True  # the code line counts as the terms being said


# --- V5-7: never the same thing twice in one call --------------------------------------------


async def test_v5_sentence_said_earlier_in_the_call_is_dropped(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _ = await _ready(api_client, make_debtor, app_engine)
    first = "Let's look at what could work for you this month. What do you think?"
    assert await session.guard_and_send(first, 1, 1) == first
    again = "Let's look at what could work for you this month! Is there a day that suits you?"
    assert await session.guard_and_send(again, 1, 1) == "Is there a day that suits you?"


async def test_v5_reply_with_nothing_new_goes_back_once(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _ = await _ready(api_client, make_debtor, app_engine)
    asked: list[str] = []

    async def retry(feedback: str) -> None:
        asked.append(feedback)

    session.request_retry = retry
    line = "Let's look at what could work for you this month."
    assert await session.guard_and_send(line, 1, 1) == line
    assert await session.guard_and_send(line, 1, 1) == ""  # nothing new: rewrite asked
    assert len(asked) == 1 and "already said that" in asked[0]
    assert await session.guard_and_send(line, 1, 1) == line  # never silent after the rewrite


async def test_v5_short_lines_and_the_scripted_opening_count_correctly(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _ = await _ready(api_client, make_debtor, app_engine)
    assert await session.guard_and_send("Does that work for you?", 1, 1) == (
        "Does that work for you?"
    )
    assert await session.guard_and_send("Does that work for you?", 1, 1) == (
        "Does that work for you?"  # short questions may repeat
    )
    opening = "Your balance with us is ¥100,000. How are things going with it?"
    said = await session.guard_and_send(f"{opening} Any questions?", 1, 1)
    # The scripted question counts as already said; a sentence with a figure may repeat.
    assert said == "Your balance with us is ¥100,000. Any questions?"


async def test_v4_never_drops_repeats(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, h = await _session(api_client, make_debtor, app_engine, "v4")
    await _verify(session, h)
    line = "Let's look at what could work for you this month."
    assert await session.guard_and_send(line, 1, 1) == line
    assert await session.guard_and_send(line, 1, 1) == line


async def test_v5_stock_prefix_before_a_comma_is_dropped(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _ = await _ready(api_client, make_debtor, app_engine)
    said = await session.guard_and_send("I understand, that amount feels high right now.", 1, 1)
    assert said == "That amount feels high right now."


async def test_v5_yes_must_follow_the_terms_themselves(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    """V5-7 (call 222): an "OK" after an unrelated question is not a promise yet."""
    session, h = await _ready(api_client, make_debtor, app_engine)
    await _offer(h, "first_offer")
    standard = await _offer(h, "other")
    await session.guard_and_send(_said(standard["terms"]), None, None)
    await session.guard_and_send("Could you tell me what monthly amount you could afford?", 1, 1)
    out, _ = await h["record_promise"]({"terms_id": standard["terms_id"]}, FM)
    assert out["recorded"] is False and "say the plan back" in out["instruction"]
    assert session.outcome is None
    await session.guard_and_send(_said(standard["terms"]), None, None)  # restated: now fine
    done, _ = await h["record_promise"]({"terms_id": standard["terms_id"]}, FM)
    assert done["recorded"] is True


def test_v5_identify_never_reads_details_back_v4_unchanged() -> None:
    from app.agent.nodes import identify_node

    rule = "Never repeat their name or date of birth back"
    assert rule in identify_node("Taro Example", V5).task
    assert rule not in identify_node("Taro Example", V4).task


# --- Voice call 6: the total, and a handoff that explains itself ------------------------------


async def test_v5_plan_offer_carries_the_total(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    _, h = await _ready(api_client, make_debtor, app_engine)
    await _offer(h, "first_offer")
    standard = await _offer(h, "other")
    assert standard["terms"]["total_to_pay"] == "¥100,000"


async def test_v5_ladder_handoff_explains_why_and_what_next(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _ = await _ready(api_client, make_debtor, app_engine)
    session.text_chat = False  # voice: the scripted goodbye is said
    h = make_handlers(session)
    await _offer(h, "first_offer")
    await _offer(h, "monthly_too_high", affordable_monthly_jpy=1_000)
    assert session.outcome == Outcome.HANDOFF.value
    assert await _last_agent_lines(app_engine, session.conversation_id, 2) == [
        scripts.without_goodbye(scripts.BELOW_FLOOR_END_V5),
        scripts.WRAP_UP_QUESTION,  # wrap_up_question: one last question before the end
    ]


async def _last_agent_lines(engine: AsyncEngine, conversation_id: int, n: int) -> list[str]:
    async with engine.connect() as conn:
        rows = await conn.scalars(
            select(Turn.text)
            .where(Turn.conversation_id == conversation_id)
            .order_by(Turn.seq.desc())
            .limit(n)
        )
        return list(reversed(list(rows)))


# --- wrap_up_question (owner request, 2026-11-03) --------------------------------------------


async def _handed_off(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> tuple[ChatSession, dict[str, Handler]]:
    session, h = await _ready(api_client, make_debtor, app_engine)
    session.text_chat = False
    h = make_handlers(session)
    await _offer(h, "first_offer")
    await _offer(h, "monthly_too_high", affordable_monthly_jpy=1_000)
    return session, h


async def test_v5_any_reply_after_anything_else_gets_a_scripted_close(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _ = await _handed_off(api_client, make_debtor, app_engine)
    assert session.wrap_outcome == Outcome.HANDOFF.value and not session.ended.is_set()
    await session.wrap_reply("Actually, can I pay 2,000 a month instead?")
    assert session.ended.is_set() and session.outcome == Outcome.HANDOFF.value
    lines = await _last_agent_lines(app_engine, session.conversation_id, 2)
    assert lines == [
        "Actually, can I pay 2,000 a month instead?",  # recorded, never negotiated
        scripts.WRAP_UP_CLOSE["handoff"],
    ]


async def test_v5_silence_after_anything_else_closes_the_call(
    api_client: httpx.AsyncClient,
    make_debtor: MakeDebtor,
    app_engine: AsyncEngine,
    monkeypatch: Any,
) -> None:
    import asyncio

    import app.agent.session as session_module

    monkeypatch.setattr(session_module, "WRAP_UP_WAIT_VOICE_S", 0.05)
    session, _ = await _handed_off(api_client, make_debtor, app_engine)
    await asyncio.wait_for(session.ended.wait(), timeout=2)
    assert (await _last_agent_lines(app_engine, session.conversation_id, 1)) == [
        scripts.WRAP_UP_SILENT_CLOSE
    ]


async def test_stop_contact_and_v4_never_ask_anything_else(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, h = await _ready(api_client, make_debtor, app_engine)
    await h["record_stop_contact"]({}, FM)
    assert session.ended.is_set() and session.wrap_outcome is None
    v4, _, h4 = await _session(api_client, make_debtor, app_engine, "v4")
    await _verify(v4, h4)
    await h4["request_handoff"]({"reason": "dispute", "summary": "Says it is not theirs."}, FM)
    assert v4.wrap_outcome is None


async def test_v5_text_ladder_end_is_scripted_and_names_our_team(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    """Text conv 8: counter_limit got an LLM goodbye ("someone will reach out")."""
    session, h = await _ready(api_client, make_debtor, app_engine)
    session.text_chat = True
    h = make_handlers(session)
    for reason in ("first_offer", "other", "monthly_too_high", "monthly_too_high"):
        await _offer(h, reason)
    assert session.closing_outcome is None  # no LLM goodbye
    lines = await _last_agent_lines(app_engine, session.conversation_id, 2)
    assert lines == [
        scripts.without_goodbye(scripts.NO_PLAN_FITS_END_V5),
        scripts.WRAP_UP_QUESTION,
    ]
    assert "colleague from our team" in lines[0]


# --- v5.11 text test: the wrap-up answers a plan question; one full confirmation -------------


async def _promised(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> ChatSession:
    session, h = await _ready(api_client, make_debtor, app_engine)
    await _offer(h, "first_offer")
    standard = await _offer(h, "other")
    await session.guard_and_send(_said(standard["terms"]), None, None)
    done, _ = await h["record_promise"]({"terms_id": standard["terms_id"]}, FM)
    assert done["recorded"] is True
    return session


async def test_v5_promise_is_read_back_by_code_then_one_question(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session = await _promised(api_client, make_debtor, app_engine)
    assert session.closing_outcome is None  # no LLM goodbye with its own question
    assert await _last_agent_lines(app_engine, session.conversation_id, 2) == [
        "Thank you, your plan is set. Your first payment of ¥10,000 is due October 31st, "
        "then ¥30,000 each month from November 30th to January 31st, 2027.",
        scripts.WRAP_UP_QUESTION,
    ]


async def test_v5_question_after_anything_else_gets_the_plan_read_back(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session = await _promised(api_client, make_debtor, app_engine)
    await session.wrap_reply("When do I have to start this installment payment?")
    last = (await _last_agent_lines(app_engine, session.conversation_id, 1))[0]
    assert last.startswith("Your first payment of ¥10,000 is due October 31st")
    assert last.endswith(scripts.WRAP_UP_ANSWERED_CLOSE)
    assert session.ended.is_set()


async def test_v5_thanks_after_anything_else_gets_the_plain_close(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session = await _promised(api_client, make_debtor, app_engine)
    await session.wrap_reply("No, that's all, thanks.")
    assert (await _last_agent_lines(app_engine, session.conversation_id, 1)) == [
        scripts.WRAP_UP_CLOSE["promise"]
    ]


async def test_v5_pay_in_full_offer_mentions_monthly_payments(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    _, h = await _ready(api_client, make_debtor, app_engine)
    full = await _offer(h, "first_offer")
    assert "monthly payments are also possible" in full["instruction"]
