"""Agent v4 (human tone): the LLM says approved terms in its own words, code checks the facts.

Every v4 switch is tested on v4 (new behavior) and on v3 (unchanged), allowed and blocked.
"""

from datetime import UTC, date, datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent import scripts
from app.agent.nodes import _v4_negotiate_rules, negotiate_node
from app.agent.session import ChatSession
from app.agent.tools import Handler
from app.agent.versions import load_version
from app.db.models import Event, Outcome
from app.policy.config import load_rules
from app.policy.output_check import Figure, check_terms_presented
from tests.conftest import MakeDebtor
from tests.test_agent_versions import V2_FIXES, V3_FIXES, V4_FIXES
from tests.test_chat_offline import FM, _session, _verify

RULES = load_rules()
V3, V4 = load_version("v3"), load_version("v4")
NOW = datetime(2026, 10, 24, 3, tzinfo=UTC)


def _said(terms: dict[str, Any]) -> str:
    """A natural reply that says every approved figure."""
    return (
        f"I understand. You could pay {terms['down_payment']} by {terms['down_payment_by']}, "
        f"then {terms['monthly_payments']} monthly payments of {terms['monthly_amount']} "
        f"starting {terms['first_monthly_due']}. Would that work for you?"
    )


async def _terms_rows(engine: AsyncEngine, conversation_id: int) -> list[Any]:
    async with engine.connect() as conn:
        return list(
            await conn.execute(
                select(Event.decision, Event.input)
                .where(Event.conversation_id == conversation_id, Event.rule == "terms_presented")
                .order_by(Event.id)
            )
        )


async def _approved(session: ChatSession, h: dict[str, Handler]) -> tuple[dict[str, Any], Any]:
    await _verify(session, h)
    return await h["propose_terms"]({"months": 3, "down_payment_jpy": 10_000}, FM)


# --- Version file ---------------------------------------------------------------------------


def test_v4_is_v3_plus_four_tone_fixes() -> None:
    assert all(getattr(V4.fixes, f) for f in (*V2_FIXES, *V3_FIXES, *V4_FIXES))
    assert V4.prompt_version == "prompts-v4" and V4.rules_version == V3.rules_version


def test_v3_prompts_have_no_v4_lines() -> None:
    assert _v4_negotiate_rules(V3, 10, 6) == ""
    v3_task = negotiate_node({"balance_jpy": 1}, RULES, V3).task
    v4_task = negotiate_node({"balance_jpy": 1}, RULES, V4).task
    assert "what's been going on" not in v3_task and "what's been going on" in v4_task
    assert f"{RULES.plan.max_months} months, {RULES.plan.min_down_payment_pct}% down" in v4_task


# --- Spoken terms ---------------------------------------------------------------------------


def test_spoken_dates_read_like_speech() -> None:
    said = [scripts.spoken_date(date(2026, 10, d)) for d in (1, 2, 3, 11, 12, 22, 31)]
    assert said == [
        "October 1st, 2026",
        "October 2nd, 2026",
        "October 3rd, 2026",
        "October 11th, 2026",
        "October 12th, 2026",
        "October 22nd, 2026",
        "October 31st, 2026",
    ]


# --- Pure rule: terms_presented -------------------------------------------------------------

REQUIRED = [Figure("amount", 30_000), Figure("amount", 45_000), Figure("count", 6)]


def test_terms_presented_allowed_when_every_figure_was_said() -> None:
    said = ["That's ¥30,000 down, then 6 monthly payments of 45,000 yen."]
    decision = check_terms_presented(NOW, terms_id=1, required=REQUIRED, said=said, rules=RULES)
    assert decision.allowed and decision.input["missing"] == []


def test_terms_presented_blocked_when_a_figure_is_missing() -> None:
    said = ["That's ¥30,000 down, and the rest in monthly payments."]
    decision = check_terms_presented(NOW, terms_id=1, required=REQUIRED, said=said, rules=RULES)
    assert not decision.allowed and decision.codes == ("terms_not_presented",)
    assert decision.input["missing"] == ["amount:45000", "count:6"]


def test_terms_presented_reads_a_spoken_full_payment_date() -> None:
    required = [Figure("amount", 100_000), Figure("date", date(2026, 10, 31))]
    ok = check_terms_presented(
        NOW, terms_id=1, required=required, rules=RULES,
        said=["You'd pay ¥100,000 by October 31st."],
    )  # fmt: skip
    wrong_day = check_terms_presented(
        NOW, terms_id=1, required=required, rules=RULES,
        said=["You'd pay ¥100,000 by October 30th."],
    )  # fmt: skip
    assert ok.allowed and not wrong_day.allowed


# --- llm_presents_terms ---------------------------------------------------------------------


async def test_v4_approved_terms_go_to_the_llm_with_spoken_dates(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine, "v4")
    result, nxt = await _approved(session, h)
    assert result["approved"] is True and nxt is None  # the LLM speaks next
    terms = result["terms"]
    assert (terms["down_payment"], terms["monthly_payments"], terms["monthly_amount"]) == (
        "¥10,000",
        3,
        "¥30,000",
    )
    assert terms["down_payment_by"].endswith(", 2026") and "-" not in terms["down_payment_by"]
    assert rec.said() == [scripts.VERIFIED_WARM]  # no scripted terms line


async def test_v3_approved_terms_are_still_said_by_code(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine, "v3")
    result, _ = await _approved(session, h)
    assert "terms" not in result and rec.said()[-1].startswith("A down payment of ¥10,000")


async def test_v4_promise_refused_until_every_term_was_said(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine, "v4")
    result, _ = await _approved(session, h)

    early, early_next = await h["record_promise"]({"terms_id": result["terms_id"]}, FM)
    assert early["recorded"] is False and early_next is None
    assert not session.ended.is_set()

    said = await session.guard_and_send(_said(result["terms"]), None, None)
    assert said == _said(result["terms"])  # every figure is approved: nothing blocked
    done, _ = await h["record_promise"]({"terms_id": result["terms_id"]}, FM)
    assert done["recorded"] is True and session.outcome == Outcome.PROMISE.value

    rows = await _terms_rows(app_engine, session.conversation_id)
    assert [r.decision.value for r in rows] == ["blocked", "allowed"]  # both audited
    assert rows[0].input["missing"] == ["amount:10000", "amount:30000", "count:3"]


async def test_v4_wrong_figure_is_blocked_and_cannot_become_a_promise(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine, "v4")
    result, _ = await _approved(session, h)
    wrong = _said(result["terms"]).replace("¥30,000", "¥25,000")
    said = await session.guard_and_send(wrong, None, None)
    assert "¥25,000" not in said and said.endswith(scripts.SAFE_FALLBACK_POST_V4)
    done, _ = await h["record_promise"]({"terms_id": result["terms_id"]}, FM)
    assert done["recorded"] is False  # the monthly amount was never actually said


async def test_v4_terms_said_before_a_newer_approval_do_not_count(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine, "v4")
    first, _ = await _approved(session, h)
    await session.guard_and_send(_said(first["terms"]), None, None)
    second, _ = await h["propose_terms"]({"months": 6, "down_payment_jpy": 10_000}, FM)
    done, _ = await h["record_promise"]({"terms_id": second["terms_id"]}, FM)
    assert done["recorded"] is False


# --- refusal_wording ------------------------------------------------------------------------


async def test_v4_discount_refusal_asks_not_to_repeat_the_number(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    for version, expected in (("v4", True), ("v3", False)):
        session, _, h = await _session(api_client, make_debtor, app_engine, version)
        await _verify(session, h)
        refused, _ = await h["check_discount_request"]({"offered_total_jpy": 10_000}, FM)
        assert refused["approved"] is False
        assert ("cannot be reduced at all" in refused.get("instruction", "")) is expected
        full, _ = await h["check_discount_request"]({"offered_total_jpy": 100_000}, FM)
        assert full["approved"] is True and "instruction" not in full


async def test_v4_blocked_refusal_falls_back_kindly_v3_unchanged(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    line = "I'm sorry, but a 90% discount is not possible."  # D-008
    for version, fallback in (
        ("v4", scripts.SAFE_FALLBACK_POST_V4),
        ("v3", scripts.SAFE_FALLBACK_POST),
    ):
        session, _, h = await _session(api_client, make_debtor, app_engine, version)
        await _verify(session, h)
        assert await session.guard_and_send(line, None, None) == fallback
        ok = "I'm sorry, I can't reduce the balance."
        assert await session.guard_and_send(ok, None, None) == ok


# --- warm_scripts ---------------------------------------------------------------------------


async def test_v4_warm_lines_v3_unchanged(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    for version, retry in (("v4", scripts.VERIFY_RETRY_WARM), ("v3", scripts.VERIFY_RETRY)):
        session, rec, h = await _session(api_client, make_debtor, app_engine, version)
        await _verify(session, h, "1993-08-09")
        assert rec.said() == [retry]


def test_warm_retry_never_says_which_detail_failed() -> None:
    line = scripts.VERIFY_RETRY_WARM.lower()
    assert "name, first and last, and your date of birth" in line
    assert "wrong date" not in line and "wrong name" not in line


# --- monthly_counts (part of llm_presents_terms) ----------------------------------------------


def test_terms_presented_reads_count_words_before_monthly() -> None:
    said = ["A down payment of 30,000 yen, then six monthly payments of ¥45,000."]
    decision = check_terms_presented(NOW, terms_id=1, required=REQUIRED, said=said, rules=RULES)
    assert decision.allowed


async def test_v4_guard_blocks_a_wrong_count_in_words_v3_unchanged(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    wrong = "You could pay it in nine monthly payments."  # 9: outside the allowed 3-6
    right = "You could pay it in six monthly payments."
    for version, blocked in (("v4", True), ("v3", False)):
        session, _, h = await _session(api_client, make_debtor, app_engine, version)
        await _verify(session, h)
        assert (await session.guard_and_send(wrong, None, None) != wrong) is blocked
        assert await session.guard_and_send(right, None, None) == right


async def _last_due(engine: AsyncEngine, conversation_id: int) -> date:
    async with engine.connect() as conn:
        schedule = await conn.scalar(
            select(Event.input["schedule"])
            .where(Event.conversation_id == conversation_id, Event.rule == "plan_terms")
            .order_by(Event.id.desc())
            .limit(1)
        )
    assert schedule is not None
    return date.fromisoformat(schedule[-1]["due"])


async def test_v4_guard_reads_month_and_year_as_a_date_v3_unchanged(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    for version in ("v4", "v3"):
        session, _, h = await _session(api_client, make_debtor, app_engine, version)
        await _approved(session, h)
        due = await _last_due(app_engine, session.conversation_id)
        right = f"The last payment would be in {due:%B %Y}."
        wrong = f"The last payment would be in {due.replace(year=due.year + 3):%B %Y}."
        # v4: the month of an approved date is allowed, any other month-year is not.
        # v3 (unchanged): the year is still read as an unapproved amount.
        assert (await session.guard_and_send(right, None, None) == right) is (version == "v4")
        assert await session.guard_and_send(wrong, None, None) != wrong


# --- No repeated plans, refusals, no stock phrases ------------------------------------------


async def test_v4_same_plan_again_is_not_read_out_but_can_still_be_promised(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, h = await _session(api_client, make_debtor, app_engine, "v4")
    first, _ = await _approved(session, h)
    await session.guard_and_send(_said(first["terms"]), None, None)
    again, nxt = await h["propose_terms"]({"months": 3, "down_payment_jpy": 10_000}, FM)
    assert again["same_as_already_offered"] is True and "terms" not in again and nxt is None
    done, _ = await h["record_promise"]({"terms_id": again["terms_id"]}, FM)
    assert done["recorded"] is True  # the person already heard every figure


async def test_v4_a_different_plan_is_read_out_in_full(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, h = await _session(api_client, make_debtor, app_engine, "v4")
    await _approved(session, h)
    other, _ = await h["propose_terms"]({"months": 6, "down_payment_jpy": 10_000}, FM)
    assert other["terms"]["monthly_payments"] == 6 and "same_as_already_offered" not in other


async def test_v4_refused_plan_asks_not_to_repeat_figures_v3_unchanged(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    for version, expected in (("v4", True), ("v3", False)):
        session, _, h = await _session(api_client, make_debtor, app_engine, version)
        await _verify(session, h)
        refused, _ = await h["propose_terms"]({"months": 12, "down_payment_jpy": 0}, FM)
        assert refused["approved"] is False
        assert ("without repeating any figure" in refused.get("instruction", "")) is expected


def test_v4_role_has_no_stock_empathy_phrase_v3_keeps_it() -> None:
    stock = "I understand, and I'll keep this brief"
    v3_role = negotiate_node({"balance_jpy": 1}, RULES, V3).role
    v4_role = negotiate_node({"balance_jpy": 1}, RULES, V4).role
    assert stock in v3_role and stock not in v4_role


# --- retry_blocked_reply / no_repeat_offers (added 2026-10-30) ---------------------------------


async def test_v4_blocked_reply_is_rewritten_once_with_the_reason(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine, "v4")
    await _verify(session, h)
    asked: list[str] = []

    async def retry(feedback: str) -> None:
        asked.append(feedback)

    session.request_retry = retry
    sent_before = len(rec.said())
    bad = "You could pay ¥99,999 a month."
    assert await session.guard_and_send(bad, None, None) == ""  # first block: nothing said
    assert len(rec.said()) == sent_before
    assert len(asked) == 1
    assert "NOT delivered" in asked[0] and "approved terms" in asked[0] and bad in asked[0]
    # The rewrite is blocked too: the fallback is said, no second retry.
    assert await session.guard_and_send(bad, None, None) == scripts.SAFE_FALLBACK_POST_V4
    assert len(asked) == 1
    # A later block gets its own retry again.
    assert await session.guard_and_send(bad, None, None) == ""
    assert len(asked) == 2


async def test_v4_retry_ends_after_a_good_rewrite(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, h = await _session(api_client, make_debtor, app_engine, "v4")
    await _verify(session, h)
    asked: list[str] = []

    async def retry(feedback: str) -> None:
        asked.append(feedback)

    session.request_retry = retry
    assert await session.guard_and_send("Pay ¥99,999 now.", None, None) == ""
    good = "What could you manage each month?"
    assert await session.guard_and_send(good, None, None) == good
    assert await session.guard_and_send("Pay ¥99,999 now.", None, None) == ""
    assert len(asked) == 2


async def test_v3_never_retries(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, h = await _session(api_client, make_debtor, app_engine, "v3")
    await _verify(session, h)
    asked: list[str] = []

    async def retry(feedback: str) -> None:
        asked.append(feedback)

    session.request_retry = retry
    assert await session.guard_and_send("Pay ¥99,999 now.", None, None) != ""
    assert asked == []


async def test_v4_an_earlier_plan_is_not_read_out_again_and_can_be_promised(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, h = await _session(api_client, make_debtor, app_engine, "v4")
    first, _ = await _approved(session, h)  # plan A: 3 months
    await session.guard_and_send(_said(first["terms"]), None, None)
    second, _ = await h["propose_terms"]({"months": 6, "down_payment_jpy": 10_000}, FM)
    assert "terms" in second  # plan B is new: read out
    await session.guard_and_send(_said(second["terms"]), None, None)
    again, nxt = await h["propose_terms"]({"months": 3, "down_payment_jpy": 10_000}, FM)
    assert again["same_as_already_offered"] is True and "terms" not in again and nxt is None
    assert {"months": 3, "down_payment_jpy": 10_000} in again["plans_already_offered"]
    assert {"months": 6, "down_payment_jpy": 10_000} in again["plans_already_offered"]
    done, _ = await h["record_promise"]({"terms_id": again["terms_id"]}, FM)
    assert done["recorded"] is True  # plan A's terms were said when it was offered


# --- echo_relative_dates: the person's own relative date may be repeated --------------------


async def test_v4_repeats_the_persons_friday_v3_still_blocks_it(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    reply = "Thanks, so you get paid Friday."
    for version in ("v4", "v3"):
        session, _, h = await _session(api_client, make_debtor, app_engine, version)
        await _verify(session, h)
        await session.record_debtor("I get paid Friday.")
        sent = await session.guard_and_send(reply, None, None)
        assert (sent == reply) is (version == "v4")


async def test_v4_relative_date_the_person_never_said_is_still_blocked(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, h = await _session(api_client, make_debtor, app_engine, "v4")
    await _verify(session, h)
    await session.record_debtor("I get paid Friday.")
    for reply in ("Could we say next Friday?", "Could you pay next month?"):
        assert await session.guard_and_send(reply, None, None) != reply


async def test_v4_approved_full_date_still_passes_with_echo(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, h = await _session(api_client, make_debtor, app_engine, "v4")
    approved, _ = await _approved(session, h)
    await session.record_debtor("I get paid Friday.")
    reply = f"Since you get paid Friday, {_said(approved['terms'])}"
    assert await session.guard_and_send(reply, None, None) == reply
