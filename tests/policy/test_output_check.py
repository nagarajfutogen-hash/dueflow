from datetime import date

import pytest

from app.policy.config import PolicyRules
from app.policy.engine import Rule
from app.policy.output_check import (
    AllowedFigures,
    OutputCheckResult,
    check_output,
    customer_month_years,
    customer_relative_phrases,
)
from tests.policy.conftest import jst

NOW = jst(2026, 10, 24, 12)
FALLBACK = "FALLBACK."
ALLOWED = AllowedFigures(
    amounts=frozenset({100_000, 10_000, 30_000}),
    counts=frozenset({3, 4, 5, 6}),
    percents=frozenset({10, 20}),
    dates=frozenset({date(2026, 10, 31), date(2026, 11, 30), date(2026, 12, 31)}),
)


def _check(rules: PolicyRules, text: str, *, verified: bool = True) -> OutputCheckResult:
    return check_output(
        NOW, text, identity_verified=verified, allowed=ALLOWED, rules=rules, fallback=FALLBACK
    )


# --- After verification: allowed figures pass -----------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Your balance is ¥100,000.",
        "A down payment of ¥10,000 by 2026-10-31, then 3 monthly installments of ¥30,000.",
        "That is 30,000 yen per month for three months.",
        "We need at least 10% down.",
        "We need at least ten percent down.",
        "The first installment is due on November 30.",
        "It's due on the 30th of November.",
        "Your next installment is due in November.",
        "May I confirm the plan?",  # "May" without a day is not a date
        "One moment please.",  # bare number word ignored
        "Thanks for talking with me today.",
        "Thanks for your courtesy.",  # "court" needs a word boundary
        "",
    ],
)
def test_allowed_output_passes_unchanged(rules: PolicyRules, text: str) -> None:
    result = _check(rules, text)
    assert result.decision.allowed is True, result.decision.reason
    assert result.text == text


# --- After verification: anything else is blocked -------------------------------------------


@pytest.mark.parametrize(
    ("text", "code", "detail"),
    [
        ("We could do ¥25,000 a month.", "unapproved_figure", "amount:25000"),
        ("How about 5,000 yen?", "unapproved_figure", "amount:5000"),
        ("You offered ¥5,000 a month, so that works.", "unapproved_figure", "amount:5000"),
        ("Let's spread it over 12 months.", "unapproved_figure", "count:12"),
        ("Let's spread it over twelve months.", "unapproved_figure", "count:12"),
        ("You could pay thirty thousand five hundred yen.", "unapproved_figure", "amount:30500"),
        ("Just pay half now.", "unapproved_figure", "percent:50"),
        ("We can accept 50% now.", "unapproved_figure", "percent:50"),
        ("Pay a quarter of it.", "unapproved_figure", "percent:25"),
        ("Pay by December 5.", "unapproved_figure", "month_day:(12, 5)"),
        ("Pay in February.", "unapproved_figure", "month:2"),
        ("Pay by 2026-11-15.", "unapproved_figure", "date:2026-11-15"),
        ("Could you pay tomorrow?", "relative_date", "relative_date:tomorrow"),
        ("Can you do it by Friday?", "relative_date", "relative_date:friday"),
    ],
)
def test_unapproved_figures_blocked(rules: PolicyRules, text: str, code: str, detail: str) -> None:
    result = _check(rules, text)
    assert result.decision.allowed is False
    assert result.decision.codes == (code,)
    assert result.decision.input["detail"] == detail
    assert result.text == FALLBACK


@pytest.mark.parametrize(
    "text",
    [
        "We will take legal action.",
        "This is your last chance.",
        "We may have to tell your employer.",
        "You could end up in COURT.",
    ],
)
@pytest.mark.parametrize("verified", [True, False])
def test_banned_language_always_blocked(rules: PolicyRules, text: str, verified: bool) -> None:
    result = _check(rules, text, verified=verified)
    assert result.decision.codes == ("banned_language",)


# --- Before verification --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["Is this a good time to talk?", "May I ask who I'm speaking with?", "Thank you."],
)
def test_pre_verification_neutral_text_passes(rules: PolicyRules, text: str) -> None:
    assert _check(rules, text, verified=False).decision.allowed is True


@pytest.mark.parametrize(
    "text",
    [
        "I'm calling about your overdue account.",
        "It's about a payment.",
        "This concerns your loan.",
        "We need ¥100,000.",  # allowed after verification, not before
        "Please call back at 3.",
        "It's regarding the arrears.",
        "Our collection team asked me to call.",
    ],
)
def test_pre_verification_disclosure_blocked(rules: PolicyRules, text: str) -> None:
    result = _check(rules, text, verified=False)
    assert result.decision.codes == ("pre_verification_disclosure",)


# --- Cut, drop and fallback -----------------------------------------------------------------


def test_violation_drops_that_sentence_and_everything_after(rules: PolicyRules) -> None:
    text = "Your balance is ¥100,000. We could do ¥25,000 a month. Does that work?"
    result = _check(rules, text)
    assert result.text == "Your balance is ¥100,000. FALLBACK."
    assert result.dropped_sentences == 2
    assert result.decision.input["kept"] == 1
    assert result.decision.input["dropped"] == 2
    assert result.decision.input["offending_sentence"] == "We could do ¥25,000 a month."


def test_violation_in_first_sentence_leaves_only_fallback(rules: PolicyRules) -> None:
    result = _check(rules, "Pay half now. Your balance is ¥100,000.")
    assert result.text == FALLBACK


def test_japanese_full_stop_splits_sentences(rules: PolicyRules) -> None:
    result = _check(rules, "Your balance is ¥100,000。 Pay ¥1 now.")
    assert result.text == "Your balance is ¥100,000。 FALLBACK."


def test_one_decision_records_checks_and_result(rules: PolicyRules) -> None:
    result = _check(rules, "Your balance is ¥100,000.")
    d = result.decision
    assert d.rule is Rule.OUTPUT_CHECK
    assert d.input["checks"] == ["banned_language", "figures"]
    assert d.input["sentences"] == 1
    assert d.input["rules_version"] == rules.version
    pre = _check(rules, "Hello.", verified=False).decision
    assert pre.input["checks"] == ["banned_language", "pre_verification"]


# --- Full dates (agent v2: fixes.full_date_parsing) -----------------------------------------

WITH_2027 = AllowedFigures(
    amounts=ALLOWED.amounts,
    counts=ALLOWED.counts,
    percents=ALLOWED.percents,
    dates=ALLOWED.dates | {date(2027, 3, 1)},
)


def _full(rules: PolicyRules, text: str, *, full_dates: bool) -> OutputCheckResult:
    return check_output(
        NOW,
        text,
        identity_verified=True,
        allowed=WITH_2027,
        rules=rules,
        fallback=FALLBACK,
        full_dates=full_dates,
    )


def test_v1_reads_the_year_of_an_approved_date_as_an_amount(rules: PolicyRules) -> None:
    # The baseline false alarm, kept as-is in v1 so the baseline stays reproducible.
    d = _full(rules, "The first payment is due March 1, 2027.", full_dates=False).decision
    assert not d.allowed and d.input["detail"] == "amount:2027"


@pytest.mark.parametrize(
    "text",
    [
        "The first payment is due March 1, 2027.",
        "The first payment is due on 1 March 2027.",
        "The first payment is due on the 1st of March, 2027.",
    ],
)
def test_v2_approved_full_date_is_allowed(rules: PolicyRules, text: str) -> None:
    result = _full(rules, text, full_dates=True)
    assert result.decision.allowed and result.text == text
    assert result.decision.input["full_dates"] is True


@pytest.mark.parametrize(
    ("text", "detail"),
    [
        ("The first payment is due April 1, 2027.", "date:2027-04-01"),
        ("The first payment is due March 1, 2028.", "date:2028-03-01"),  # right day, wrong year
        ("Pay by February 30, 2027.", "date:February 30, 2027"),  # not a real date
        ("Pay ¥2,027 by March 1, 2027.", "amount:2027"),  # a real amount is still checked
    ],
)
def test_v2_unapproved_full_date_is_blocked(rules: PolicyRules, text: str, detail: str) -> None:
    d = _full(rules, text, full_dates=True).decision
    assert not d.allowed and d.input["detail"] == detail


def test_v2_full_date_before_verification_is_blocked(rules: PolicyRules) -> None:
    d = check_output(
        NOW,
        "Call us before March 1, 2027.",
        identity_verified=False,
        allowed=WITH_2027,
        rules=rules,
        fallback=FALLBACK,
        full_dates=True,
    ).decision
    assert not d.allowed and d.codes == ("pre_verification_disclosure",)


# --- Agent v4 echo_relative_dates: relative dates only in the person's own words ------------


def _echo(rules: PolicyRules, text: str, customer_said: list[str]) -> OutputCheckResult:
    return check_output(
        NOW,
        text,
        identity_verified=True,
        allowed=ALLOWED,
        rules=rules,
        fallback=FALLBACK,
        full_dates=True,  # as agent v4 checks
        monthly_counts=True,
        customer_relative=customer_relative_phrases(customer_said, rules),
    )


@pytest.mark.parametrize(
    ("customer", "agent"),
    [
        ("I get paid Friday.", "Got it, you get paid Friday."),
        ("I can pay next month.", "Next month works better for you, I understand."),
        ("Payday is on friday", "So FRIDAY is your payday."),
    ],
)
def test_echo_of_the_persons_own_relative_date_is_allowed(
    rules: PolicyRules, customer: str, agent: str
) -> None:
    result = _echo(rules, agent, [customer])
    assert result.decision.allowed
    assert result.text == agent


@pytest.mark.parametrize(
    ("customer", "agent", "detail"),
    [
        ("I get paid Friday.", "We could follow up next Friday.", "relative_date:next friday"),
        ("I get paid Friday.", "Could you pay next month?", "relative_date:next month"),
        ("I'm busy today.", "Could you pay tomorrow?", "relative_date:tomorrow"),
        ("I can pay next month.", "Next month, or maybe Monday?", "relative_date:monday"),
    ],
)
def test_relative_date_the_person_did_not_say_is_blocked(
    rules: PolicyRules, customer: str, agent: str, detail: str
) -> None:
    result = _echo(rules, agent, [customer])
    assert not result.decision.allowed
    assert result.decision.codes == ("relative_date",)
    assert result.decision.input["detail"] == detail
    assert result.text == FALLBACK


def test_echo_keeps_checking_every_other_figure(rules: PolicyRules) -> None:
    ok = "You get paid Friday, so a down payment of ¥10,000 by October 31st, 2026 could work."
    assert _echo(rules, ok, ["I get paid Friday"]).decision.allowed
    bad = "You get paid Friday, so maybe ¥5,000 then."
    result = _echo(rules, bad, ["I get paid Friday"])
    assert result.decision.codes == ("unapproved_figure",)


def test_echo_records_the_persons_relative_dates_and_off_means_never(
    rules: PolicyRules,
) -> None:
    on = _echo(rules, "Friday it is.", ["paid Friday"])
    assert on.decision.input["customer_relative_dates"] == ["friday"]
    off = _check(rules, "Friday it is.")  # switch off (v1-v3): never allowed, no new key
    assert not off.decision.allowed
    assert "customer_relative_dates" not in off.decision.input


# --- V5-5: a month and year the person said may be repeated (agent v5 echo_month_year) ------


def _months(rules: PolicyRules, text: str, customer_said: list[str]) -> OutputCheckResult:
    return check_output(
        NOW,
        text,
        identity_verified=True,
        allowed=ALLOWED,
        rules=rules,
        fallback=FALLBACK,
        full_dates=True,
        monthly_counts=True,
        customer_months=customer_month_years(customer_said),
    )


def test_month_year_the_person_said_is_allowed(rules: PolicyRules) -> None:
    result = _months(rules, "May 2027 is a long way off.", ["I could pay in May 2027."])
    assert result.decision.allowed
    assert result.decision.input["customer_month_years"] == ["2027-05"]


def test_month_year_the_person_did_not_say_is_blocked(rules: PolicyRules) -> None:
    result = _months(rules, "How about June 2027?", ["I could pay in May 2027."])
    assert not result.decision.allowed
    assert result.decision.codes == ("unapproved_figure",)
    off = check_output(  # switch off (v4): blocked, no new key
        NOW,
        "May 2027 is far.",
        identity_verified=True,
        allowed=ALLOWED,
        rules=rules,
        fallback=FALLBACK,
        full_dates=True,
        monthly_counts=True,
    )
    assert not off.decision.allowed
    assert "customer_month_years" not in off.decision.input
