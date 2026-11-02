from datetime import date

import pytest

from app.policy.config import PolicyRules
from app.policy.engine import Decision, Rule, check_identity
from app.policy.identity import find_dates, name_in_text, normalize_name
from tests.policy.conftest import jst

NOW = jst(2026, 10, 24, 12)
NAME = "Taro Example"
DOB = date(1985, 4, 12)


@pytest.mark.parametrize(
    "claimed",
    [
        "Taro Example",
        "taro example",
        "  TARO   EXAMPLE ",
        "Example Taro",  # Japanese order: family name first
        "Example, Taro",
        "Taro-Example",
        "Ｔａｒｏ　Ｅｘａｍｐｌｅ",  # full-width letters and space
    ],
)
def test_name_normalization_matches(claimed: str) -> None:
    assert normalize_name(claimed) == normalize_name(NAME)


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("My name is Taro Example.", True),
        ("example taro here", True),
        ("This is Taro.", False),
        ("Taro Exampel", False),
        ("", False),
    ],
)
def test_name_in_text(text: str, found: bool) -> None:
    assert name_in_text(NAME, text) is found


@pytest.mark.parametrize(
    "text",
    [
        "I was born on 1985-04-12.",
        "born 1985/4/12",
        "DOB 1985.04.12 thanks",
        "it's 19850412",
        "1985年4月12日です",
        "１９８５－０４－１２",  # full-width
        "April 12, 1985",
        "born april 12 1985",
        "Apr 12th 1985",
        "on 12 April 1985",
        "the 12th of April, 1985",
    ],
)
def test_dates_found_in_sentences(text: str) -> None:
    assert find_dates(text) == {DOB}


@pytest.mark.parametrize(
    "text",
    ["04/12/1985", "12/04/1985", "1985-13-01", "1985-02-30", "yesterday", "", "Smarch 12 1985"],
)
def test_ambiguous_or_invalid_dates_not_found(text: str) -> None:
    assert find_dates(text) == set()


def test_phone_like_numbers_are_not_dates() -> None:
    assert find_dates("call 0120-000-000 or 090-1234-5678") == set()


def _check(rules: PolicyRules, text: str, failed: int = 0) -> Decision:
    return check_identity(
        NOW,
        debtor_text=text,
        actual_name=NAME,
        actual_dob=DOB,
        failed_attempts=failed,
        rules=rules,
    )


def test_identity_verified_from_own_words(rules: PolicyRules) -> None:
    d = _check(rules, "Yes. I'm Example Taro, born April 12, 1985.")
    assert d.rule is Rule.IDENTITY_VERIFICATION
    assert (d.allowed, d.codes) == (True, ())


@pytest.mark.parametrize(
    "text",
    ["Yes, this is Taro Example.", "Speaking.", "Born 04/12/1985, Taro Example."],
)
def test_no_date_stated_is_not_an_attempt(rules: PolicyRules, text: str) -> None:
    d = _check(rules, text)
    assert d.allowed is False
    assert d.codes == ("no_details",)
    assert d.input["attempt"] is None


def test_wrong_name_and_wrong_dob_give_identical_reason(rules: PolicyRules) -> None:
    wrong_name = _check(rules, "Jiro Example, 1985-04-12")
    wrong_dob = _check(rules, "Taro Example, 1985-04-13")
    assert not wrong_name.allowed and not wrong_dob.allowed
    assert wrong_name.reason == wrong_dob.reason
    assert wrong_name.codes == wrong_dob.codes == ("identity_mismatch",)


def test_second_failure_reaches_max(rules: PolicyRules) -> None:
    d = _check(rules, "Nobody, 1970-01-01", failed=rules.identity.max_failed_attempts - 1)
    assert d.codes == ("identity_mismatch", "max_attempts_reached")


def test_exhausted_attempts_block_without_comparing(rules: PolicyRules) -> None:
    d = _check(rules, "Taro Example, 1985-04-12", failed=rules.identity.max_failed_attempts)
    assert (d.allowed, d.codes) == (False, ("max_attempts_reached",))
    assert "name_matched" not in d.input


def test_identity_input_never_contains_pii(rules: PolicyRules) -> None:
    for text in ("Taro Example 1985-04-12", "Secret Guess 1970-01-01", "just Taro"):
        d = _check(rules, text)
        logged = str(d.input) + d.reason
        for pii in ("Taro", "Example", "1985", "Secret", "1970"):
            assert pii not in logged


# --- V5-7 voice: dates said in words (live call 3: STT left "... twentieth nineteen eighty
# four" as words, so identity never read it) ------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("April twentieth nineteen eighty four.", date(1984, 4, 20)),
        ("april twenty nineteen eighty-four", date(1984, 4, 20)),
        ("the twenty-first of March, nineteen ninety", date(1990, 3, 21)),
        ("first of May nineteen oh five", date(1905, 5, 1)),
        ("June 3rd two thousand and one", date(2001, 6, 3)),
        ("December thirty first two thousand", date(2000, 12, 31)),
        ("It's Taro Example, born August eighth nineteen ninety three.", date(1993, 8, 8)),
    ],
)
def test_dates_said_in_words_are_read(text: str, expected: date) -> None:
    assert find_dates(text) == {expected}


@pytest.mark.parametrize(
    "text",
    [
        "wait a second, I can pay one thousand yen",
        "April nineteen eighty four",  # no day: ambiguous, never guessed
        "February thirtieth nineteen eighty",  # not a real date
        "twenty twenty",
    ],
)
def test_spoken_numbers_that_are_not_a_full_date_are_not_read(text: str) -> None:
    assert find_dates(text) == set()


def test_a_date_split_over_two_voice_turns_is_read_when_joined() -> None:
    """V5-7 (live call 4): "April twelfth." and "1985." arrived as two turns."""
    assert find_dates(" ".join(["April twelfth.", "1985."])) == {date(1985, 4, 12)}
    assert find_dates("the twelfth of April. 1985") == {date(1985, 4, 12)}
    assert find_dates("April twelfth.") == set()  # alone: no year, never guessed
