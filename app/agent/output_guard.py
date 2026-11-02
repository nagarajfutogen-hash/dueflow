"""Runs the output interceptor on an LLM reply with figures known to code, and records it."""

import re
from datetime import date, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncConnection

from app.agent import scripts
from app.agent.actions import payments_on_file
from app.agent.conversation import OpenConversation, open_conversation, record
from app.agent.identity import is_verified
from app.agent.negotiation import TERMS_RULES
from app.agent.versions import load_version
from app.db.models import (
    Callback,
    Decision,
    Event,
    EventType,
    Payment,
    Promise,
    PromiseStatus,
    Speaker,
    Turn,
)
from app.policy.config import PolicyRules
from app.policy.engine import Decision as PolicyDecision
from app.policy.output_check import (
    SENTENCE_SPLIT,
    AllowedFigures,
    OutputCheckResult,
    check_output,
    customer_month_years,
    customer_relative_phrases,
)


async def allowed_figures(
    conn: AsyncConnection,
    convo: OpenConversation,
    rules: PolicyRules,
    *,
    payments_on_record: bool = False,
    callbacks: bool = False,
) -> AllowedFigures:
    """Figures the agent may say: balance, current history, latest approved terms, rule limits.
    Never figures the debtor offered: those must go through propose_terms first."""
    amounts: set[int] = {convo.balance_jpy}
    counts: set[int] = set(range(rules.plan.min_months, rules.plan.max_months + 1))
    counts.add(rules.pay_in_full.max_days_until_payment)
    percents: set[int] = {
        rules.plan.min_down_payment_pct,
        rules.plan.min_down_payment_pct_after_broken_promise,
    }
    dates: set[date] = set()

    promises = await conn.execute(
        select(
            Promise.months,
            Promise.down_payment_jpy,
            Promise.installment_jpy,
            Promise.first_due_date,
        ).where(Promise.debtor_id == convo.debtor_id, Promise.status != PromiseStatus.SUPERSEDED)
    )
    for p in promises:
        amounts |= {p.down_payment_jpy, p.installment_jpy}
        counts.add(p.months)
        dates.add(p.first_due_date)
    paid = await conn.scalar(
        select(func.sum(Payment.amount_jpy)).where(Payment.debtor_id == convo.debtor_id)
    )
    if paid:
        amounts.add(int(paid))
    if payments_on_record:  # v4 payment_check: each payment's amount and date
        for pay in await payments_on_file(conn, convo.debtor_id, convo.timezone):
            amounts.add(pay.amount_jpy)
            dates.add(pay.paid_on)

    if callbacks:  # v5: the callback date asked for in this call, or that this call fulfils
        dates |= set(
            await conn.scalars(
                select(Callback.due_date).where(
                    (Callback.conversation_id == convo.conversation_id)
                    | (Callback.done_conversation_id == convo.conversation_id)
                )
            )
        )

    latest_terms = await conn.scalar(
        select(Event.input)
        .where(
            Event.conversation_id == convo.conversation_id,
            Event.type == EventType.POLICY_DECISION,
            Event.rule.in_(TERMS_RULES),
            Event.decision == Decision.ALLOWED,
        )
        .order_by(Event.id.desc())
        .limit(1)
    )
    if latest_terms:
        schedule = latest_terms["schedule"]
        amounts |= {int(p["amount_jpy"]) for p in schedule}
        dates |= {date.fromisoformat(p["due"]) for p in schedule}
        counts.add(sum(1 for p in schedule if p["kind"] == "installment"))
    amounts.discard(0)
    return AllowedFigures(
        frozenset(amounts), frozenset(counts), frozenset(percents), frozenset(dates)
    )


async def guard_agent_reply(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    text: str,
    now: datetime,
    rules: PolicyRules,
    agent_version: str,
) -> str:
    """Check one LLM-written reply, record one output_check event, return what may be said."""
    result = await check_agent_reply(
        conn,
        conversation_id=conversation_id,
        text=text,
        now=now,
        rules=rules,
        agent_version=agent_version,
    )
    return result.text


_RETRY_HINTS = {
    "unapproved_figure": "It had a figure that is not in the approved terms. Use only the "
    "figures from your account facts or latest tool result, exactly as written; never "
    "calculate a new amount yourself; say every date in full (month, day and year).",
    "relative_date": "It had a relative date. Say exact dates in full (month, day and year); "
    "you may repeat a relative day only in the person's own words (for example 'Friday' if "
    "they said 'Friday'), never one they did not say.",
    "banned_language": "It had a phrase that is not allowed. Say it calmly, with no threats, "
    "legal words or pressure.",
    "unverified_claim": "It said something about interest, fees or charges that you have no "
    "data for. Never say there is or is not interest or a fee: say you don't have that detail "
    "and a colleague can confirm.",
    "pre_verification_disclosure": "It mentioned the account before identity was verified. "
    "Do not mention the account, amounts or dates yet.",
}


# v5 (V5-6): a whole first sentence that is only a stock acknowledgement. The prompt alone did
# not stop them ("Thanks for letting me know." in calls 208, 213, 215).
_STOCK_OPENER = re.compile(
    r"^(?:(?:thanks|thank you)(?: so much| very much)?(?: for (?:letting me know|sharing(?: that)?"
    r"|telling me(?: that)?|that|explaining(?: that)?|being (?:open|honest)(?: about that)?"
    r"|your (?:time|patience|honesty)))?"
    r"|i (?:understand|see|hear you|appreciate (?:that|it|you sharing that|your (?:time|honesty)))"
    r"|i appreciate you (?:letting me know|sharing that))[.!]$",
    re.I,
)


# v5 (V5-6): the agent asking what they can pay (calls 185, 205, 208 asked it again and again
# instead of calling next_offer).
_AMOUNT_ASK = re.compile(
    r"\b(?:how much|what amount|what (?:monthly )?payment|the most you|any amount"
    r"|what (?:could|can|would) you (?:manage|afford|pay))\b[^?]*\?",
    re.I,
)
AMOUNT_ASKED_AGAIN = (
    "Your last reply was NOT delivered: you already asked what they can pay for this option. "
    "Do not ask again. Call next_offer now with reason 'other' (or the reason they gave), "
    "and an amount only if they said one."
)


def asks_for_amount(text: str) -> bool:
    return bool(_AMOUNT_ASK.search(text))


# V5-7: "I understand, that amount feels high..." (call 218): the stock words before a comma.
_STOCK_PREFIX = re.compile(r"^(?:i understand|i see|i hear you|got it|i get it),\s+(?=\w)", re.I)


def strip_stock_opener(text: str) -> str:
    """Drop a first sentence that is only a stock acknowledgement, when more follows, and a
    stock 'I understand, ' in front of the first sentence. Only removes words (the next word
    gets a capital letter); never adds or changes any."""
    sentences = [s for s in SENTENCE_SPLIT.split(text.strip()) if s]
    if len(sentences) > 1 and _STOCK_OPENER.match(sentences[0].strip()):
        sentences = sentences[1:]
    first = _STOCK_PREFIX.sub("", sentences[0]) if sentences else ""
    if first and first != sentences[0]:
        sentences[0] = first[0].upper() + first[1:]
    return " ".join(sentences) if sentences else text


# V5-7: a sentence said earlier in the call is not said again. Short ones ("Does that work
# for you?") may repeat: asking again is normal.
REPEAT_MIN_WORDS = 6
_NON_WORD = re.compile(r"[^\w¥]+")
SAID_ALREADY = (
    "Your last reply was NOT delivered: you already said that earlier in this call. Say "
    "something new that moves the conversation on, in your own words; never repeat yourself."
)


def sentence_key(sentence: str) -> str | None:
    """A sentence for the repeat check (case and punctuation ignored); None if it is short or
    holds a figure: saying terms again ("say that again?", confirming before a promise) is
    allowed."""
    if any(c.isdigit() or c == "¥" for c in sentence):
        return None
    words = _NON_WORD.sub(" ", sentence.lower()).split()
    return " ".join(words) if len(words) >= REPEAT_MIN_WORDS else None


def said_keys(text: str) -> set[str]:
    keys = (sentence_key(s) for s in SENTENCE_SPLIT.split(text.strip()) if s)
    return {k for k in keys if k}


def drop_repeats(text: str, said: set[str]) -> str:
    """The reply without sentences already said in this call ('' if nothing new is left)."""
    kept = [
        s for s in SENTENCE_SPLIT.split(text.strip()) if s and (sentence_key(s) or "") not in said
    ]
    return " ".join(kept)


def retry_feedback(decision: PolicyDecision) -> str:
    """v4 (fixes.retry_blocked_reply): why a reply was not sent, for the LLM to rewrite it.
    Names the problem and the blocked sentence (the LLM's own words), never debtor data."""
    code = decision.codes[0] if decision.codes else ""
    hint = _RETRY_HINTS.get(code, "It did not pass the safety check.")
    sentence = decision.input.get("offending_sentence", "")
    return (
        f"Your last reply was NOT delivered to the person. {hint} Blocked part: "
        f'"{sentence}" ({decision.input.get("detail", code)}). Write the reply again now, '
        "keeping it short and natural; do not mention this check."
    )


async def check_agent_reply(
    conn: AsyncConnection,
    *,
    conversation_id: int,
    text: str,
    now: datetime,
    rules: PolicyRules,
    agent_version: str,
    goodbye: bool = False,
) -> OutputCheckResult:
    """Check one LLM-written reply and record one output_check event. goodbye: the v4
    closing line, written after the outcome was recorded (the call has ended)."""
    convo = await open_conversation(conn, conversation_id, allow_ended=goodbye)
    verified = await is_verified(conn, conversation_id)
    fixes = load_version(agent_version).fixes
    post = scripts.SAFE_FALLBACK_POST_V4 if fixes.refusal_wording else scripts.SAFE_FALLBACK_POST
    if fixes.one_option_at_a_time:  # v5 (V5-5): fits any moment of the call
        post = scripts.SAFE_FALLBACK_POST_V5
    customer_relative = None
    customer_months = None
    if fixes.echo_relative_dates or fixes.echo_month_year:
        said = list(
            await conn.scalars(
                select(Turn.text).where(
                    Turn.conversation_id == conversation_id, Turn.speaker == Speaker.DEBTOR
                )
            )
        )
        if fixes.echo_relative_dates:  # v4: relative dates from the person's own words only
            customer_relative = customer_relative_phrases(said, rules)
        if fixes.echo_month_year:  # v5 (V5-5): a month and year the person said
            customer_months = customer_month_years(said)
    result = check_output(
        now,
        text,
        identity_verified=verified,
        allowed=await allowed_figures(
            conn,
            convo,
            rules,
            payments_on_record=fixes.payment_check,
            callbacks=fixes.callbacks,
        ),
        rules=rules,
        fallback=post if verified else scripts.SAFE_FALLBACK_PRE,
        full_dates=fixes.full_date_parsing,
        monthly_counts=fixes.llm_presents_terms,
        customer_relative=customer_relative,
        claims_check=fixes.no_interest_claims,
        customer_months=customer_months,
    )
    await record(conn, convo, result.decision, agent_version)
    return result
