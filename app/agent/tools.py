"""Tool handlers: the LLM proposes by calling these, code decides.

Scripted lines (verification results, approved terms, goodbyes) are said by code here,
verbatim; the LLM is then kept silent with NO_RESPONSE so it cannot paraphrase them.
"""

from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

from pipecat.flows import NO_RESPONSE, FlowManager, FlowsFunctionSchema, NodeConfig
from pipecat.flows.types import FlowArgs

from app.agent import scripts
from app.agent.actions import (
    check_payment_claim_record,
    record_no_workplace_contact,
    record_stop_contact,
    request_handoff,
)
from app.agent.callbacks import record_callback
from app.agent.conversation import ConversationEndedError, open_conversation
from app.agent.history import get_account_history
from app.agent.identity import (
    ThirdPartyKind,
    VerificationStatus,
    confirm_wrong_number,
    handle_third_party,
    verify_identity,
)
from app.agent.negotiation import (
    LadderOutcome,
    TermsNotApprovedError,
    check_discount_request,
    ladder_next_offer,
    ladder_propose_terms,
    propose_pay_in_full,
    propose_terms,
    record_promise,
    terms_schedule,
    terms_were_said,
)
from app.agent.nodes import APPROACHES, NodeSpec, negotiate_node
from app.agent.session import ChatSession
from app.db.audit import record_tool_call
from app.db.models import HandoffReason, Outcome
from app.policy.engine import PaymentClaim
from app.policy.ladder import Action, Rung

ToolResult = tuple[dict[str, Any], Any]  # (result for the LLM, next node | None | NO_RESPONSE)
Handler = Callable[[FlowArgs, FlowManager], Awaitable[ToolResult]]

ASK_ACCEPT = "Does that work for you?"
# v4 llm_goodbyes: seconds to wait for the LLM's goodbye before the scripted one is said.
GOODBYE_WAIT_S = 20.0
_GOODBYE = (
    "The call is ending now: {what} Say a short, warm goodbye in your own words, in one or two "
    "sentences that fit this conversation. No questions, no new figures or dates."
)
# v5 (V5-5): every V5-4 goodbye was "Thank you... Take care, and have a good day!"
_GOODBYE_V5 = (
    " Keep it under 20 words and specific to this call. No 'Thank you for...' opener, no "
    "'Take care, and have a good day', no 'Wishing you all the best', no 'feel free to reach "
    "out'. Say dates like "
    "'November 24th' (the year only if it is not this year)."
)
_GOODBYE_WHAT = {
    HandoffReason.HARDSHIP: "a colleague will contact them to look at their situation.",
    HandoffReason.DISPUTE: "a colleague will look into what they said; do not talk about payment.",
    HandoffReason.UNRESOLVED: "a colleague will contact them.",
    HandoffReason.BELOW_ACCOUNT_FLOOR: "a colleague will contact them to look at their situation "
    "and what is possible. Do not mention any amount, plan or date.",
}
# v5 payment_ladder: the handoff summary for each ladder escalation, written by code.
_LADDER_SUMMARY = {
    "below_account_floor": "The customer said they cannot pay what the lowest allowed plan needs.",
    "counter_limit": "The customer did not accept the allowed options offered in this call.",
    "no_offer_left": "No allowed option is left for this customer in this call.",
}


def _opt_int(value: Any) -> int | None:
    """An optional positive whole-yen amount from a tool argument; anything else is None."""
    try:
        amount = int(value)
    except (TypeError, ValueError):
        return None
    return amount if amount > 0 else None


_PAYMENT_CHECK_WHAT = (
    "a colleague will check the payment they described. Thank them for the details and say "
    "it may simply still be processing; never say they are wrong."
)
# v4 payment_check: claims with no match before the dispute handoff (one chance to correct).
PAYMENT_CLAIM_TRIES = 2
_PROMISE_WHAT = (
    "their plan is recorded. Thank them; you may repeat the first payment amount and its "
    "date exactly as you said them."
)


def to_flows_node(spec: NodeSpec, handlers: dict[str, Handler]) -> NodeConfig:
    """Adapter: framework-free NodeSpec -> Pipecat Flows NodeConfig."""
    return NodeConfig(
        name=spec.name,
        role_message=spec.role,
        task_messages=[{"role": "developer", "content": spec.task}],
        respond_immediately=spec.respond_immediately,
        functions=[
            FlowsFunctionSchema(
                name=tool.name,
                description=tool.description,
                properties=tool.properties,
                required=list(tool.required),
                handler=handlers[tool.name],
            )
            for tool in spec.tools
        ],
    )


def _plans(offered: dict[tuple[Any, ...], list[str]]) -> list[dict[str, int]]:
    """Plans offered so far, as months + down payment (what propose_terms takes)."""
    return [
        {
            "months": sum(1 for p in plan if p.kind == "installment"),
            "down_payment_jpy": sum(p.amount_jpy for p in plan if p.kind == "down_payment"),
        }
        for plan in offered
    ]


def make_handlers(session: ChatSession) -> dict[str, Handler]:
    s = session
    handlers: dict[str, Handler] = {}

    async def _end_with(say: str, outcome: Outcome, result: dict[str, Any]) -> ToolResult:
        if s.wraps_up(outcome.value):
            say = scripts.without_goodbye(say)  # v5: "anything else?" follows
        await s.say_scripted(say)
        await s.end(outcome.value)
        return result, NO_RESPONSE

    def _this_year() -> int | None:
        """v5 (V5-6): dates for the LLM drop this year ('November 24th'); the model copied
        'November 24th, 2026' word for word. v1-v4: None (always the full date)."""
        return s.clock.now().year if s.version.fixes.one_option_at_a_time else None

    async def _goodbye(say: str, outcome: Outcome, result: dict[str, Any], what: str) -> ToolResult:
        """v4 (fixes.llm_goodbyes, text chat): the LLM says the goodbye (guarded); otherwise,
        or if it is blocked or late, the scripted line. The outcome is already recorded."""
        if not (s.version.fixes.llm_goodbyes and s.text_chat):
            return await _end_with(say, outcome, result)
        s.start_closing(outcome.value, say, GOODBYE_WAIT_S)
        instruction = _GOODBYE.format(what=what)
        if s.version.fixes.one_option_at_a_time:
            instruction += _GOODBYE_V5
        if s.wraps_up(outcome.value):
            instruction += (
                " Don't ask any question and don't say goodbye: the system asks one last "
                "question next."
            )
        return {**result, "instruction": instruction}, None

    async def verify(args: FlowArgs, fm: FlowManager) -> ToolResult:
        async with s.engine.begin() as conn:
            result = await verify_identity(
                conn,
                conversation_id=s.conversation_id,
                debtor_text=" ".join(s.identity_words),
                now=s.clock.now(),
                rules=s.rules,
                agent_version=s.agent_version,
            )
        await s.tool_result("verify_identity", {"status": result.status.value})
        if result.status is VerificationStatus.NEED_DETAILS:
            # Not an attempt: the person has not stated a date of birth yet.
            return {
                "status": "need_details",
                "instruction": "Ask for their full name and date of birth, for example "
                "'April 12, 1990' or '1990-04-12'. Do not call verify_identity until they say it.",
            }, None
        s.identity_words.clear()
        if result.status is VerificationStatus.RETRY:
            await s.say_scripted(scripts.VERIFY_RETRY)
            return {"status": "retry", "attempts_left": result.attempts_left}, NO_RESPONSE
        if result.status is VerificationStatus.FAILED_FINAL:
            await s.say_scripted(scripts.VERIFY_FAILED_END)
            await s.end(Outcome.END_WRONG_PERSON.value)
            return {"status": "failed"}, NO_RESPONSE

        await s.say_scripted(scripts.VERIFIED)
        # LoadHistory runs in code, not as an LLM choice.
        async with s.engine.begin() as conn:
            history = await get_account_history(
                conn,
                conversation_id=s.conversation_id,
                now=s.clock.now(),
                agent_version=s.agent_version,
            )
        facts = history.facts or {}
        if facts.get("open_dispute"):
            async with s.engine.begin() as conn:
                say = await request_handoff(
                    conn,
                    conversation_id=s.conversation_id,
                    reason=HandoffReason.DISPUTE,
                    summary="Open dispute on file; negotiation blocked until resolved.",
                    now=s.clock.now(),
                    agent_version=s.agent_version,
                )
            if s.version.fixes.warm_scripts:
                say = scripts.OPEN_DISPUTE_END_WARM  # v4: says why the call ends
            return await _end_with(say, Outcome.HANDOFF, {"status": "handoff"})
        s.node = "negotiate"
        if s.version.fixes.one_option_at_a_time and "balance_jpy" in facts:
            # V5-6: code says the opening, so the LLM first speaks after the customer has
            # (calls 207, 212: its first reply repeated case notes nobody had mentioned).
            await s.say_scripted(scripts.v5_opening(int(facts["balance_jpy"])))
        return {"status": "verified"}, to_flows_node(
            negotiate_node(facts, s.rules, s.version), handlers
        )

    async def third_party(args: FlowArgs, fm: FlowManager) -> ToolResult:
        kind = ThirdPartyKind(str(args.get("kind", "wrong_number")))
        if kind is ThirdPartyKind.WRONG_NUMBER and s.version.fixes.verified_wrong_number:
            async with s.engine.begin() as conn:
                confirmed = await confirm_wrong_number(
                    conn,
                    conversation_id=s.conversation_id,
                    debtor_text=" ".join(s.debtor_words),
                    now=s.clock.now(),
                    rules=s.rules,
                    agent_version=s.agent_version,
                )
            if not confirmed:
                out = {
                    "status": "not_confirmed",
                    "instruction": "The person has not said this is a wrong number. Do not end "
                    f"the call. Politely ask whether you are speaking with {s.full_name}.",
                }
                await s.tool_result("report_third_party", out)
                return out, None
        async with s.engine.begin() as conn:
            say = await handle_third_party(
                conn,
                conversation_id=s.conversation_id,
                kind=kind,
                now=s.clock.now(),
                agent_version=s.agent_version,
            )
        outcome = (
            Outcome.THIRD_PARTY_CALLBACK
            if kind is ThirdPartyKind.NOT_AVAILABLE
            else Outcome.END_WRONG_PERSON
        )
        await s.say_scripted(say)
        await s.end(outcome.value)
        return {"status": "ended"}, NO_RESPONSE

    async def stop_contact(args: FlowArgs, fm: FlowManager) -> ToolResult:
        async with s.engine.begin() as conn:
            say = await record_stop_contact(
                conn,
                conversation_id=s.conversation_id,
                now=s.clock.now(),
                agent_version=s.agent_version,
            )
        return await _end_with(say, Outcome.STOP_CONTACT, {"status": "ended"})

    async def no_workplace(args: FlowArgs, fm: FlowManager) -> ToolResult:
        async with s.engine.begin() as conn:
            say = await record_no_workplace_contact(
                conn,
                conversation_id=s.conversation_id,
                now=s.clock.now(),
                agent_version=s.agent_version,
            )
        await s.say_scripted(say)
        return {"status": "recorded"}, NO_RESPONSE

    async def _proposal_result(name: str, proposal: Any) -> ToolResult:
        if proposal.approved and proposal.terms and s.version.fixes.llm_presents_terms:
            # v4: the LLM says the approved terms in its own words, through the interceptor.
            earlier = (
                s.version.fixes.no_repeat_offers
                and proposal.schedule != s.offered_schedule
                and proposal.schedule in s.offered_plans
            )
            if earlier:
                # A plan offered earlier in the call: not read out again. Its terms were said
                # back then, so that saying counts again for record_promise.
                if s.offered_schedule:
                    s.offered_plans[s.offered_schedule] = list(s.said_since_terms)
                s.offered_schedule = proposal.schedule
                s.said_since_terms = list(s.offered_plans[proposal.schedule])
                out = {
                    "approved": True,
                    "terms_id": proposal.terms_id,
                    "same_as_already_offered": True,
                    "plans_already_offered": _plans(s.offered_plans),
                    "instruction": "You already offered this exact plan earlier in the call. "
                    "Do not read it out again. If they now want it, call record_promise with "
                    "this terms_id. Otherwise propose a plan not in plans_already_offered (for "
                    "example more months or a smaller down payment), or if none is left, say "
                    "kindly that a colleague can look at their situation and call "
                    "request_handoff(hardship).",
                }
                await s.tool_result(name, {"approved": True, "same_as_already_offered": True})
                return out, None
            if proposal.schedule == s.offered_schedule:
                # The same plan again (the LLM re-proposed it): never read it out a second time.
                # Nothing is cleared, so the earlier saying still counts for record_promise.
                out = {
                    "approved": True,
                    "terms_id": proposal.terms_id,
                    "same_as_already_offered": True,
                    "instruction": "This is exactly the plan you already offered. Do not read "
                    "it out again. If they want it, call record_promise with this terms_id. If it "
                    "is too much for them, say kindly it is the lowest you can offer and that a "
                    "colleague can look at their situation, then call request_handoff(hardship).",
                }
                await s.tool_result(name, {"approved": True, "same_as_already_offered": True})
                return out, None
            if s.offered_schedule:
                s.offered_plans[s.offered_schedule] = list(s.said_since_terms)
            s.offered_schedule = proposal.schedule
            s.offered_plans.setdefault(proposal.schedule, [])
            s.said_since_terms.clear()
            out = {
                "approved": True,
                "terms_id": proposal.terms_id,
                "terms": scripts.terms_facts(list(proposal.schedule), _this_year()),
                "instruction": "Offer these terms warmly in your own words as an option for "
                "them (not as already approved or decided), using every figure exactly as "
                "written (amounts like ¥6,000, dates, and the number of monthly payments), "
                "then ask whether it works for them.",
            }
            await s.tool_result(name, {"approved": True, "terms_id": proposal.terms_id})
            return out, None
        if proposal.approved and proposal.terms:
            await s.tool_result(name, {"approved": True, "terms_id": proposal.terms_id})
            await s.say_scripted(f"{proposal.terms} {ASK_ACCEPT}")
            return {
                "approved": True,
                "terms_id": proposal.terms_id,
                "already_said_to_debtor": proposal.terms,
            }, NO_RESPONSE
        result = {
            "approved": False,
            "refusal_codes": list(proposal.refusal_codes),
            "reason": proposal.refusal_reason,
        }
        await s.tool_result(name, result)
        if proposal.handoff_required:
            async with s.engine.begin() as conn:
                say = await request_handoff(
                    conn,
                    conversation_id=s.conversation_id,
                    reason=HandoffReason.DISPUTE,
                    summary="Open dispute found during negotiation.",
                    now=s.clock.now(),
                    agent_version=s.agent_version,
                )
            return await _goodbye(
                say, Outcome.HANDOFF, result, _GOODBYE_WHAT[HandoffReason.DISPUTE]
            )
        if s.version.fixes.refusal_wording:
            result["instruction"] = (
                "Say kindly that this is not possible, without repeating any figure from the "
                "reason or from what they offered."
                + (
                    " Remind them of the plan you already offered, without reading it all out "
                    "again, or ask what they could manage."
                    if s.offered_schedule
                    else " Ask what they could manage, then check an allowed plan."
                )
            )
        return result, None  # the LLM explains the refusal (through the interceptor)

    async def _ladder_result(name: str, outcome: LadderOutcome) -> ToolResult:
        """v5: an engine offer is presented like v4 approved terms; no option left -> the
        system hands off, with a summary written by code. below_account_floor only when the
        customer SAID an amount below every plan (PLANNING §6 rule 3); counter_limit and
        no_offer_left (every option refused) are stored as unresolved (VPS audit, conv 8)."""
        if outcome.action is Action.ESCALATE:
            code = outcome.escalate or "no_offer_left"
            reason = (
                HandoffReason.BELOW_ACCOUNT_FLOOR
                if code == "below_account_floor"
                else HandoffReason.UNRESOLVED
            )
            await s.tool_result(name, {"escalate": code})
            async with s.engine.begin() as conn:
                say = await request_handoff(
                    conn,
                    conversation_id=s.conversation_id,
                    reason=reason,
                    summary=_LADDER_SUMMARY.get(code, _LADDER_SUMMARY["no_offer_left"]),
                    now=s.clock.now(),
                    agent_version=s.agent_version,
                )
            if s.version.fixes.one_option_at_a_time:
                # Voice call 6 / text conv 8: explain why and what happens next, in fixed
                # professional words for every ladder end (the LLM wrote "someone will reach
                # out"), in text and voice alike.
                say = (
                    scripts.BELOW_FLOOR_END_V5
                    if code == "below_account_floor"
                    else scripts.NO_PLAN_FITS_END_V5
                )
                return await _end_with(
                    say, Outcome.HANDOFF, {"status": "ended", "handoff": reason.value}
                )
            return await _goodbye(
                say,
                Outcome.HANDOFF,
                {"status": "ended", "handoff": reason.value},
                _GOODBYE_WHAT[HandoffReason.BELOW_ACCOUNT_FLOOR],
            )
        assert outcome.proposal is not None
        out, nxt = await _proposal_result(name, outcome.proposal)
        if outcome.proposal.approved and outcome.rung is not None:
            out["option"] = outcome.rung.value
            if outcome.rung is Rung.PAY_IN_FULL and s.version.fixes.one_option_at_a_time:
                # v5.11 text test: the prompt rule alone never said plans exist too.
                out["instruction"] = (
                    str(out.get("instruction", ""))
                    + " Then add in a few words that monthly payments are also possible if "
                    "that's easier."
                )
            if outcome.refused_codes:
                out["their_plan_possible"] = False
                out["instruction"] = (
                    "Their own plan is not possible: say so in a few kind words without "
                    "repeating any of its figures, then " + str(out.get("instruction", ""))
                )
        return out, nxt

    async def callback(args: FlowArgs, fm: FlowManager) -> ToolResult:
        """v5 (fixes.callbacks): a callback on a day the customer named, resolved by code."""
        async with s.engine.begin() as conn:
            result = await record_callback(
                conn,
                conversation_id=s.conversation_id,
                day_phrase=str(args.get("day_phrase", "")),
                reason=str(args.get("reason", "other")),
                now=s.clock.now(),
                rules=s.rules,
                agent_version=s.agent_version,
            )
        if not result.recorded or result.due is None:
            # V5-5: no relative day in the wording (the guard blocks 'tomorrow' the customer
            # did not say, call 192), and the reason matches the code.
            if "too_far" in result.codes:
                why = "Say kindly that a call back can only be booked within the next few weeks."
            elif "too_soon" in result.codes:
                why = "Say kindly that a call back can't be booked for today."
            else:
                why = "That day could not be worked out from their words."
            out = {
                "recorded": False,
                "codes": list(result.codes),
                "instruction": f"{why} In one short question, ask which day suits them for a "
                "call back; never suggest a day yourself.",
            }
            await s.tool_result("record_callback", out)
            return out, None
        await s.tool_result("record_callback", {"recorded": True, "due": result.due.isoformat()})
        when = scripts.spoken_date(result.due, _this_year())
        return await _goodbye(
            result.say,
            Outcome.CALLBACK_SCHEDULED,
            {"recorded": True, "callback_on": when},
            f"their request for a call back on {when} is noted and a colleague will confirm "
            "it. You may say that date exactly as written. Do not talk about payment.",
        )

    async def next_offer_tool(args: FlowArgs, fm: FlowManager) -> ToolResult:
        """v5 (fixes.payment_ladder): the engine's next option down its ladder."""
        async with s.engine.begin() as conn:
            outcome = await ladder_next_offer(
                conn,
                conversation_id=s.conversation_id,
                reason=str(args.get("reason", "other"))[:40],
                affordable_monthly_jpy=_opt_int(args.get("affordable_monthly_jpy")),
                affordable_down_jpy=_opt_int(args.get("affordable_down_jpy")),
                now=s.clock.now(),
                rules=s.rules,
                agent_version=s.agent_version,
            )
        return await _ladder_result("next_offer", outcome)

    async def terms(args: FlowArgs, fm: FlowManager) -> ToolResult:
        if s.version.fixes.payment_ladder:
            async with s.engine.begin() as conn:
                outcome = await ladder_propose_terms(
                    conn,
                    conversation_id=s.conversation_id,
                    months=int(args.get("months", 0)),
                    down_payment_jpy=int(args.get("down_payment_jpy", 0)),
                    now=s.clock.now(),
                    rules=s.rules,
                    agent_version=s.agent_version,
                )
            return await _ladder_result("propose_terms", outcome)
        async with s.engine.begin() as conn:
            proposal = await propose_terms(
                conn,
                conversation_id=s.conversation_id,
                months=int(args.get("months", 0)),
                down_payment_jpy=int(args.get("down_payment_jpy", 0)),
                now=s.clock.now(),
                rules=s.rules,
                agent_version=s.agent_version,
            )
        return await _proposal_result("propose_terms", proposal)

    async def pay_in_full(args: FlowArgs, fm: FlowManager) -> ToolResult:
        if s.version.fixes.payment_ladder:
            # V5-5: the customer's words for the day; code resolves the date (call 198).
            async with s.engine.begin() as conn:
                balance = (await open_conversation(conn, s.conversation_id)).balance_jpy
                proposal = await propose_pay_in_full(
                    conn,
                    conversation_id=s.conversation_id,
                    amount_jpy=balance,
                    day_phrase=str(args.get("day_phrase", ""))[:60],
                    now=s.clock.now(),
                    rules=s.rules,
                    agent_version=s.agent_version,
                )
            out, nxt = await _proposal_result("propose_pay_in_full", proposal)
            if proposal.approved and "terms" in out:
                # V5-7 (call 219): their own request, not an option the agent picked.
                out["instruction"] = (
                    "Their own request works: confirm it back in a few words with the amount "
                    "and date exactly as written (not as 'one option'), then ask if they want "
                    "to go ahead."
                )
            if "date_not_said" in proposal.refusal_codes:
                out["instruction"] = (
                    "That day could not be worked out from their words. In one short question, "
                    "ask which day they could make the full payment; never suggest a day "
                    "yourself. If they have no day in mind, call next_offer(reason='other')."
                )
            return out, nxt
        try:
            when = date.fromisoformat(str(args.get("payment_date", "")))
        except ValueError:
            return {"approved": False, "reason": "payment_date must be YYYY-MM-DD"}, None
        async with s.engine.begin() as conn:
            balance = (await open_conversation(conn, s.conversation_id)).balance_jpy
            proposal = await propose_pay_in_full(
                conn,
                conversation_id=s.conversation_id,
                amount_jpy=balance,
                payment_date=when,
                now=s.clock.now(),
                rules=s.rules,
                agent_version=s.agent_version,
            )
        return await _proposal_result("propose_pay_in_full", proposal)

    async def discount(args: FlowArgs, fm: FlowManager) -> ToolResult:
        async with s.engine.begin() as conn:
            result = await check_discount_request(
                conn,
                conversation_id=s.conversation_id,
                offered_total_jpy=int(args.get("offered_total_jpy", 0)),
                now=s.clock.now(),
                agent_version=s.agent_version,
                reject_zero_offer=s.version.fixes.payment_claim_handoff,
            )
        out: dict[str, Any] = {"approved": result.approved, "reason": result.refusal_reason}
        if not result.approved and s.version.fixes.refusal_wording:
            out["instruction"] = (
                "Say kindly and plainly that the balance cannot be reduced at all (do not say "
                "their offer is too low, and do not repeat their number or percentage), then "
                "offer to find a payment plan that fits instead."
            )
        if "invalid_offer" in result.refusal_codes:
            out["instruction"] = (
                "This is not a discount request. If the person says the debt is already paid, "
                "call request_handoff with reason dispute."
            )
        await s.tool_result("check_discount_request", out)
        return out, None

    async def promise(args: FlowArgs, fm: FlowManager) -> ToolResult:
        terms_id = int(args.get("terms_id", 0))
        if s.version.fixes.llm_presents_terms:
            # v5 (V5-7, call 222): the "yes" must answer the terms themselves, so the agent's
            # LAST line must hold every figure; v4 accepts any line since the terms.
            v5_confirm = s.version.fixes.one_option_at_a_time
            heard = s.said_since_terms[-1:] if v5_confirm else s.said_since_terms
            async with s.engine.begin() as conn:
                said = await terms_were_said(
                    conn,
                    conversation_id=s.conversation_id,
                    terms_id=terms_id,
                    said=heard,
                    now=s.clock.now(),
                    rules=s.rules,
                    agent_version=s.agent_version,
                )
            if not said:
                out = {
                    "recorded": False,
                    "reason": "the person has not heard all the approved terms yet",
                    "instruction": "Say every approved figure clearly (down payment and date, "
                    "monthly amount, number of monthly payments), ask them to confirm, and "
                    "only then call record_promise again.",
                }
                if v5_confirm:
                    out["instruction"] = (
                        "Their yes did not follow the plan itself. In one sentence, say the plan "
                        "back with every figure (down payment and date, monthly amount, number "
                        "of monthly payments; or the full amount and date) and ask if they want "
                        "to go ahead. Call record_promise only after they clearly say yes."
                    )
                await s.tool_result("record_promise", out)
                return out, None
        try:
            async with s.engine.begin() as conn:
                say = await record_promise(
                    conn,
                    conversation_id=s.conversation_id,
                    terms_id=terms_id,
                    now=s.clock.now(),
                    agent_version=s.agent_version,
                )
        except TermsNotApprovedError:
            out = {"recorded": False, "reason": "terms_id is not the latest approved terms"}
            await s.tool_result("record_promise", out)
            return out, None
        await s.tool_result("record_promise", {"recorded": True})
        if s.wraps_up(Outcome.PROMISE.value):
            # v5 (text test, v5.11): the LLM confirmation named only the first payment and
            # asked its own question. Code reads back the whole plan; "anything else?" follows.
            async with s.engine.connect() as conn:
                schedule = await terms_schedule(conn, terms_id)
            s.promised_schedule = tuple(schedule)
            readback = scripts.plan_readback(schedule, _this_year())
            return await _end_with(
                f"Thank you, your plan is set. {readback}", Outcome.PROMISE, {"recorded": True}
            )
        return await _goodbye(say, Outcome.PROMISE, {"recorded": True}, _PROMISE_WHAT)

    async def handoff(args: FlowArgs, fm: FlowManager) -> ToolResult:
        reason_raw = str(args.get("reason", "unresolved"))
        reason = (
            HandoffReason(reason_raw)
            if reason_raw in {r.value for r in HandoffReason}
            else HandoffReason.UNRESOLVED
        )
        if (
            s.version.fixes.payment_ladder
            and reason is HandoffReason.UNRESOLVED
            and not s.unresolved_handoff_checked
        ):
            # V5-5 (call 205): the LLM handed off as 'unresolved' with ladder options left.
            # Affordability belongs to the engine; a real 'unresolved' goes through next time.
            s.unresolved_handoff_checked = True
            out = {
                "handed_off": False,
                "instruction": "Not handed off yet. If this is about what they can afford or "
                "an option not working for them, call next_offer instead (the system hands off "
                "when no option is left). Only if it is something else, call request_handoff "
                "again.",
            }
            await s.tool_result("request_handoff", out)
            return out, None
        async with s.engine.begin() as conn:
            say = await request_handoff(
                conn,
                conversation_id=s.conversation_id,
                reason=reason,
                summary=str(args.get("summary", "")),
                now=s.clock.now(),
                agent_version=s.agent_version,
            )
        what = _GOODBYE_WHAT.get(reason, _GOODBYE_WHAT[HandoffReason.UNRESOLVED])
        return await _goodbye(say, Outcome.HANDOFF, {"status": "ended"}, what)

    async def payment_claim(args: FlowArgs, fm: FlowManager) -> ToolResult:
        """v4 (fixes.payment_check): a "paid" claim checked against payments on record."""
        try:
            paid_on = date.fromisoformat(str(args["paid_on"])) if args.get("paid_on") else None
        except ValueError:
            paid_on = None
        amount = args.get("amount_jpy")
        claim = PaymentClaim(
            paid_on=paid_on,
            amount_jpy=int(amount) if amount not in (None, "") else None,
            method=str(args["method"])[:32] if args.get("method") else None,
            reference=str(args["reference"])[:64] if args.get("reference") else None,
        )
        async with s.engine.begin() as conn:
            decision, balance = await check_payment_claim_record(
                conn,
                conversation_id=s.conversation_id,
                claim=claim,
                now=s.clock.now(),
                agent_version=s.agent_version,
            )
        await s.tool_result("check_payment_claim", {"found": decision.allowed})
        if decision.allowed:
            pay = decision.input["payment"]
            found = {
                "amount_jpy": pay["amount_jpy"],
                "paid_on": scripts.spoken_date(date.fromisoformat(pay["paid_on"])),
                "method": pay["method"],
            }
            if decision.input["match"] == "amount_only":
                return {
                    "found": "same_amount_other_date",
                    "payment_on_record": found,
                    "balance_still_owed_jpy": balance,
                    "instruction": "Say kindly you can see a payment of that amount on the "
                    "date on record (say it exactly as here) and ask if that is the one they "
                    "mean. Do not say they are wrong.",
                }, None
            return {
                "found": True,
                "payment_on_record": found,
                # Already net of every payment on record: the LLM must not subtract.
                "balance_still_owed_jpy": balance,
                "instruction": "Thank them and confirm that payment (amount and date exactly "
                "as here). Then say balance_still_owed_jpy exactly (it already includes this "
                "payment; never subtract anything) and ask how they would like to "
                "handle it. If they say they paid more than this, ask for the details of the "
                "other payment and check it the same way.",
            }, None
        if "claim_details_missing" in decision.codes:
            return {
                "found": False,
                "instruction": "Ask when they paid, how much, how, and the transaction ID or "
                "receipt number if they have one, then check again.",
            }, None
        s.payment_claims_unmatched += 1
        if s.payment_claims_unmatched < PAYMENT_CLAIM_TRIES:
            return {
                "found": False,
                "instruction": "Say kindly you can't see that payment on record yet; it may "
                "still be processing. Ask whether any detail might be different (the date, "
                "the amount or the transaction ID), then check again with what they say. Do "
                "not repeat their figures and never say they are wrong.",
            }, None
        c = decision.input["claim"]
        summary = (
            "Says the debt is paid; no matching payment on record. They said: "
            f"date {c['paid_on'] or '-'}, amount {c['amount_jpy'] or '-'}, "
            f"method {c['method'] or '-'}, reference {c['reference'] or '-'}."
        )
        async with s.engine.begin() as conn:
            say = await request_handoff(
                conn,
                conversation_id=s.conversation_id,
                reason=HandoffReason.DISPUTE,
                summary=summary,
                now=s.clock.now(),
                agent_version=s.agent_version,
            )
        return await _goodbye(say, Outcome.HANDOFF, {"found": False}, _PAYMENT_CHECK_WHAT)

    async def set_approach(args: FlowArgs, fm: FlowManager) -> ToolResult:
        """v4 (fixes.approaches): tone and order of questions only; audited, no policy."""
        approach = str(args.get("approach", ""))
        if approach not in APPROACHES:
            return {"set": False, "allowed": list(APPROACHES)}, None
        reason = str(args.get("reason", ""))[:200]
        async with s.engine.begin() as conn:
            convo = await open_conversation(conn, s.conversation_id)
            await record_tool_call(
                conn,
                conversation_id=s.conversation_id,
                debtor_id=convo.debtor_id,
                tool="set_approach",
                at=s.clock.now(),
                agent_version=s.agent_version,
                detail={"approach": approach, "reason": reason},
            )
        s.approach = approach
        await s.tool_result("set_approach", {"approach": approach})
        return {"set": True, "approach": approach, "how": APPROACHES[approach]}, None

    def _safe(fn: Handler) -> Handler:
        """A tool called after the call ended (late LLM output) does nothing."""

        async def wrapper(args: FlowArgs, fm: FlowManager) -> ToolResult:
            if s.ended.is_set():
                return {"status": "call already ended"}, NO_RESPONSE
            if s.closing_outcome is not None:
                # v4: a tool call instead of the goodbye: say the scripted goodbye.
                await s.close_scripted()
                return {"status": "call already ended"}, NO_RESPONSE
            try:
                return await fn(args, fm)
            except ConversationEndedError:
                return {"status": "call already ended"}, NO_RESPONSE

        return wrapper

    handlers.update(
        {
            "verify_identity": _safe(verify),
            "report_third_party": _safe(third_party),
            "record_stop_contact": _safe(stop_contact),
            "record_no_workplace_contact": _safe(no_workplace),
            "propose_terms": _safe(terms),
            "next_offer": _safe(next_offer_tool),
            "record_callback": _safe(callback),
            "propose_pay_in_full": _safe(pay_in_full),
            "check_discount_request": _safe(discount),
            "record_promise": _safe(promise),
            "request_handoff": _safe(handoff),
            "set_approach": _safe(set_approach),
            "check_payment_claim": _safe(payment_claim),
        }
    )
    return handlers
