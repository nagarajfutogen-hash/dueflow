"""Fixed agent lines, rendered in code (never by the LLM).

Before identity is verified, nothing here may reveal the debt or the purpose of the call
(Art. 21 third-party disclosure). tests/test_identity.py enforces that.
"""

import re
from datetime import date

from app.policy.schedule import Payment

SCRIPTS_VERSION = "scripts-v1"
# Before verification only the neutral name is used: "Servicing" could reveal the
# business type to a third party (Art. 21). The full name is given once verified.
NEUTRAL_COMPANY_NAME = "Dueflow"
COMPANY_NAME = "Dueflow Servicing (demo)"
CALLBACK_NUMBER = "0120-000-000 (demo)"


def opening(full_name: str) -> str:
    """First agent message: AI + company disclosure, then ask for the debtor by name."""
    return (
        f"Hello, this is an AI assistant calling on behalf of {NEUTRAL_COMPANY_NAME}. "
        f"May I speak with {full_name}?"
    )


VERIFY_PROMPT = "Before we continue, please confirm your full name and your date of birth."

# Identical whichever field failed: never reveal which one was wrong.
VERIFY_RETRY = (
    "Sorry, I wasn't able to verify those details. "
    "Please tell me your full name and your date of birth again."
)

VERIFY_FAILED_END = (
    "I'm sorry, I wasn't able to verify your identity, so I can't continue this call. Goodbye."
)

# Never repeats the date of birth back. First line allowed to name the full company.
VERIFIED = f"Thank you, your identity is confirmed. I'm calling from {COMPANY_NAME}."


def third_party_callback(full_name: str) -> str:
    """Neutral: caller identity + callback request only. No purpose, no debt."""
    return (
        f"Thank you. Could you please ask {full_name} to call {NEUTRAL_COMPANY_NAME} back "
        f"at {CALLBACK_NUMBER}? Goodbye."
    )


# Identify node turn cap reached without a stated date of birth. Neutral: no purpose, no debt.
NO_VERIFICATION_END = (
    "I'm sorry, I can't continue without confirming who I'm speaking with. Goodbye."
)

WRONG_NUMBER_END = "I'm sorry for the trouble, I must have the wrong number. Goodbye."

# Safe fallbacks appended by the output interceptor after a blocked sentence.
SAFE_FALLBACK_PRE = (
    "Sorry, let me start again. Before we continue, "
    "please confirm your full name and your date of birth."
)
SAFE_FALLBACK_POST = "Sorry, I misspoke. Let me check the exact details before we continue."
# v4 (fixes.refusal_wording): a blocked reply after verification is usually a refusal that
# repeated the person's figure (D-008); this reads as a refusal, not a slip.
SAFE_FALLBACK_POST_V4 = "I'm sorry, I can't offer that. Let's find a plan that works for you."
# v5 (V5-5): the v4 line read as a refusal even after a callback or a goodbye (calls 192, 196).
SAFE_FALLBACK_POST_V5 = "Sorry, I put that badly. What would work best for you?"

# v4 (fixes.warm_scripts): kinder wording, same content. Still code text, never the LLM.
# The retry still never says which detail failed.
VERIFY_RETRY_WARM = (
    "Hmm, I couldn't match those details. Could you tell me your full name, first and last, "
    "and your date of birth once more?"
)

PRE_VERIFICATION_LINES = (
    VERIFY_PROMPT,
    VERIFY_RETRY,
    VERIFY_RETRY_WARM,
    VERIFY_FAILED_END,
    WRONG_NUMBER_END,
    NO_VERIFICATION_END,
    SAFE_FALLBACK_PRE,
)


def yen(amount: int) -> str:
    return f"¥{amount:,}"


def terms_text(schedule: list[Payment]) -> str:
    """Canonical terms, rendered from the approved schedule. The agent says exactly this."""
    if len(schedule) == 1:
        return f"One full payment of {yen(schedule[0].amount_jpy)} by {schedule[0].due}."
    down, installments = schedule[0], schedule[1:]
    first, last = installments[0], installments[-1]
    monthly = f"{len(installments)} monthly installments of {yen(first.amount_jpy)}"
    if last.amount_jpy != first.amount_jpy:
        monthly += f" (the last one {yen(last.amount_jpy)})"
    return (
        f"A down payment of {yen(down.amount_jpy)} by {down.due}, then {monthly}, "
        f"from {first.due} to {last.due}."
    )


def v5_opening(balance_jpy: int) -> str:
    """Agent v5 (V5-6): said by code right after verification; the LLM replies to the answer."""
    return f"Your balance with us is {yen(balance_jpy)}. How are things going with it?"


def offer_line(schedule: list[Payment], this_year: int | None = None) -> str:
    """Agent v5 (V5-6): the current offer in plain spoken words, said by code when the LLM's
    reply presenting it was blocked twice (call 215: the offer was never heard)."""
    if len(schedule) == 1:
        return (
            f"You could pay the full {yen(schedule[0].amount_jpy)} in one payment by "
            f"{spoken_date(schedule[0].due, this_year)}. Does that work for you?"
        )
    down, installments = schedule[0], schedule[1:]
    first, last = installments[0], installments[-1]
    monthly = f"{len(installments)} monthly payments of {yen(first.amount_jpy)}"
    if last.amount_jpy != first.amount_jpy:
        monthly += f" (the last one {yen(last.amount_jpy)})"
    return (
        f"You could pay {yen(down.amount_jpy)} by {spoken_date(down.due, this_year)}, then "
        f"{monthly}, starting {spoken_date(first.due, this_year)}. Does that work for you?"
    )


def plan_readback(schedule: list[Payment], this_year: int | None = None) -> str:
    """v5: the recorded plan in plain spoken words, built by code from the stored schedule."""
    if len(schedule) == 1:
        p = schedule[0]
        return f"Your payment of {yen(p.amount_jpy)} is due by {spoken_date(p.due, this_year)}."
    down, installments = schedule[0], schedule[1:]
    first, last = installments[0], installments[-1]
    text = (
        f"Your first payment of {yen(down.amount_jpy)} is due {spoken_date(down.due, this_year)}, "
        f"then {yen(first.amount_jpy)} each month from {spoken_date(first.due, this_year)} to "
        f"{spoken_date(last.due, this_year)}."
    )
    if last.amount_jpy != first.amount_jpy:
        text += f" The last payment is {yen(last.amount_jpy)}."
    return text


def promise_confirmed(terms: str) -> str:
    return f"Thank you. I've recorded your agreement: {terms}"


STOP_CONTACT_END = "Understood. We will not contact you again. Goodbye."
NO_WORKPLACE_ACK = "Understood. We will not contact you at your workplace."
HANDOFF_END = "I'll pass this to a member of our team, who will contact you. Goodbye."

# v5 (fixes.wrap_up_question): one last question before the call ends, then a scripted close.
WRAP_UP_QUESTION = "Is there anything else I can help you with today?"
WRAP_UP_CLOSE = {
    "promise": "You're welcome. Thanks again, and take care. Goodbye.",
    "handoff": "A colleague will contact you about that. Take care, goodbye.",
    "callback_scheduled": "We'll talk then. Take care, goodbye.",
}
WRAP_UP_ANSWERED_CLOSE = "If anything else comes up, our team can help. Take care, goodbye."
WRAP_UP_SILENT_CLOSE = "Thank you for your time today. Goodbye."
_GOODBYE_TAIL = re.compile(r"(?:,\s*|\s+)(?:goodbye|bye)[.!]?\s*$", re.I)


def without_goodbye(text: str) -> str:
    """'..., take care, goodbye.' -> '..., take care.': the call goes on to one last question."""
    return _GOODBYE_TAIL.sub(".", text.rstrip())


# v5 (voice call 6): the ladder handoff ended abruptly; say why, and what happens next.
BELOW_FLOOR_END_V5 = (
    "I'm sorry, I can't set up a plan at that amount today. A colleague from our team will "
    "contact you to look at what's possible for you. Thank you for talking with me, goodbye."
)
# v5 (text conv 8): every allowed option was offered and none suits them (counter_limit,
# no_offer_left).
NO_PLAN_FITS_END_V5 = (
    "I'm sorry we couldn't find a plan that works for you today. A colleague from our team "
    "will contact you to look at what's possible for you. Thank you for talking with me, "
    "goodbye."
)
VERIFIED_WARM = f"Thank you, that's confirmed. I'm calling from {COMPANY_NAME}."
HANDOFF_END_WARM = (
    "I'll pass this to a member of our team, and they'll be in touch with you. Take care, goodbye."
)
# v4: an open dispute on file ends the call right after verification; give the context
# (why we called, what is on file) before saying why the call ends.
OPEN_DISPUTE_END_WARM = (
    "I'm calling about your loan account with us. I can see you raised a concern about this "
    "account, and our team is reviewing it now. Until that review is finished, we won't ask "
    "you for any payment. The colleague handling it will contact you to go through it with "
    "you. Thank you for your patience. Take care, goodbye."
)


def callback_end(due: date) -> str:
    """v5 callback goodbye: the request is noted; the owner still has to approve the call."""
    return (
        f"Thank you. I've noted that you'd like a call back on {spoken_date(due)}, and a "
        "colleague will confirm it. Take care, goodbye."
    )


STOP_CONTACT_END_WARM = "Understood. We won't contact you again. Take care, goodbye."
WARM = {
    VERIFY_RETRY: VERIFY_RETRY_WARM,
    VERIFIED: VERIFIED_WARM,
    HANDOFF_END: HANDOFF_END_WARM,
    STOP_CONTACT_END: STOP_CONTACT_END_WARM,
}


def spoken_date(d: date, this_year: int | None = None) -> str:
    """'October 31st, 2026': how a person says a date (the guard reads this form).
    this_year (agent v5, V5-6): 'October 31st' when the date is in that year, as people say it
    (the guard matches a month and day to the approved date)."""
    suffix = "th" if 11 <= d.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d.day % 10, "th")
    if d.year == this_year:
        return f"{d:%B} {d.day}{suffix}"
    return f"{d:%B} {d.day}{suffix}, {d.year}"


def terms_facts(schedule: list[Payment], this_year: int | None = None) -> dict[str, str | int]:
    """v4 (fixes.llm_presents_terms): approved terms handed to the LLM to say in its own
    words. Every figure here is also an allowed figure for the guard. this_year: see
    spoken_date (agent v5 only)."""
    if len(schedule) == 1:
        return {
            "full_payment": yen(schedule[0].amount_jpy),
            "full_payment_by": spoken_date(schedule[0].due, this_year),
        }
    down, installments = schedule[0], schedule[1:]
    facts: dict[str, str | int] = {
        "down_payment": yen(down.amount_jpy),
        "down_payment_by": spoken_date(down.due, this_year),
        "monthly_payments": len(installments),
        "monthly_amount": yen(installments[0].amount_jpy),
        "first_monthly_due": spoken_date(installments[0].due, this_year),
        "last_monthly_due": spoken_date(installments[-1].due, this_year),
    }
    if installments[-1].amount_jpy != installments[0].amount_jpy:
        facts["last_monthly_amount"] = yen(installments[-1].amount_jpy)
    if this_year is not None:  # v5 (voice call 6): asked "the total?", it never said it
        facts["total_to_pay"] = yen(sum(p.amount_jpy for p in schedule))
    return facts
