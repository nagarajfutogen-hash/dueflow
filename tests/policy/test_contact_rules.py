from datetime import UTC, datetime, timedelta

import pytest

from app.policy.config import PolicyRules
from app.policy.engine import (
    DialTarget,
    Rule,
    check_attempt_limits,
    check_contact_window,
    check_stop_contact,
    check_workplace_contact,
    pre_call_check,
)
from tests.policy.conftest import jst

NOW = jst(2026, 10, 24, 12)

# --- Contact window ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hour", "minute", "second", "allowed"),
    [
        (7, 59, 59, False),
        (8, 0, 0, True),
        (12, 0, 0, True),
        (20, 59, 0, True),
        (20, 59, 59, True),
        (21, 0, 0, False),
        (23, 0, 0, False),
        (0, 0, 0, False),
        (3, 0, 0, False),
    ],
)
def test_contact_window_edges_jst(
    rules: PolicyRules, hour: int, minute: int, second: int, allowed: bool
) -> None:
    d = check_contact_window(jst(2026, 10, 24, hour, minute, second), "Asia/Tokyo", rules)
    assert d.rule is Rule.CONTACT_WINDOW
    assert d.allowed is allowed
    assert d.codes == (() if allowed else ("outside_contact_window",))


def test_non_jst_debtor_uses_own_timezone(rules: PolicyRules) -> None:
    # 13:00 UTC = 22:00 JST (blocked in Tokyo) = 09:00 EDT (allowed in New York).
    now = datetime(2026, 10, 24, 13, tzinfo=UTC)
    assert check_contact_window(now, "Asia/Tokyo", rules).allowed is False
    assert check_contact_window(now, "America/New_York", rules).allowed is True


def test_non_jst_debtor_blocked_in_own_night(rules: PolicyRules) -> None:
    # 03:00 UTC = 12:00 JST (allowed in Tokyo) = 23:00 EDT (blocked in New York).
    now = datetime(2026, 10, 24, 3, tzinfo=UTC)
    assert check_contact_window(now, "Asia/Tokyo", rules).allowed is True
    assert check_contact_window(now, "America/New_York", rules).allowed is False


def test_dst_change_is_respected(rules: PolicyRules) -> None:
    # New York leaves DST on 2026-11-01: same UTC clock time, one hour earlier locally.
    before = datetime(2026, 10, 31, 12, 30, tzinfo=UTC)  # 08:30 EDT
    after = datetime(2026, 11, 1, 12, 30, tzinfo=UTC)  # 07:30 EST
    assert check_contact_window(before, "America/New_York", rules).allowed is True
    assert check_contact_window(after, "America/New_York", rules).allowed is False


def test_invalid_timezone_fails_closed(rules: PolicyRules) -> None:
    d = check_contact_window(NOW, "Mars/Olympus_Mons", rules)
    assert d.allowed is False
    assert d.codes == ("invalid_timezone",)


def test_naive_datetime_rejected(rules: PolicyRules) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        check_contact_window(datetime(2026, 10, 24, 12), "Asia/Tokyo", rules)


def test_contact_window_records_local_time_and_rules_version(rules: PolicyRules) -> None:
    d = check_contact_window(datetime(2026, 10, 24, 13, tzinfo=UTC), "America/New_York", rules)
    assert d.input["debtor_local_time"].startswith("2026-10-24T09:00")
    assert d.input["rules_version"] == rules.version
    assert d.at == datetime(2026, 10, 24, 13, tzinfo=UTC)


# --- Attempt limits ---------------------------------------------------------------------


def test_no_prior_attempts_allowed(rules: PolicyRules) -> None:
    d = check_attempt_limits(NOW, "Asia/Tokyo", [], rules)
    assert d.allowed is True
    assert d.codes == ()


def test_earlier_attempt_same_local_day_blocked(rules: PolicyRules) -> None:
    d = check_attempt_limits(NOW, "Asia/Tokyo", [jst(2026, 10, 24, 9)], rules)
    assert d.allowed is False
    assert "daily_limit_reached" in d.codes


def test_attempt_yesterday_allowed(rules: PolicyRules) -> None:
    d = check_attempt_limits(NOW, "Asia/Tokyo", [jst(2026, 10, 23, 20)], rules)
    assert d.allowed is True


def test_attempt_just_before_local_midnight_counts_for_previous_day(rules: PolicyRules) -> None:
    d = check_attempt_limits(jst(2026, 10, 24, 8), "Asia/Tokyo", [jst(2026, 10, 23, 23, 59)], rules)
    assert d.allowed is True


def test_two_attempts_in_week_allowed(rules: PolicyRules) -> None:
    prior = [jst(2026, 10, 20, 10), jst(2026, 10, 22, 10)]
    assert check_attempt_limits(NOW, "Asia/Tokyo", prior, rules).allowed is True


def test_three_attempts_in_week_blocked(rules: PolicyRules) -> None:
    prior = [jst(2026, 10, 19, 10), jst(2026, 10, 21, 10), jst(2026, 10, 23, 10)]
    d = check_attempt_limits(NOW, "Asia/Tokyo", prior, rules)
    assert d.allowed is False
    assert d.codes == ("weekly_limit_reached",)


def test_attempt_older_than_seven_days_drops_out(rules: PolicyRules) -> None:
    prior = [NOW - timedelta(days=7, minutes=1), jst(2026, 10, 21, 10), jst(2026, 10, 23, 10)]
    assert check_attempt_limits(NOW, "Asia/Tokyo", prior, rules).allowed is True


def test_attempt_exactly_seven_days_ago_drops_out(rules: PolicyRules) -> None:
    prior = [NOW - timedelta(days=7), jst(2026, 10, 21, 10), jst(2026, 10, 23, 10)]
    assert check_attempt_limits(NOW, "Asia/Tokyo", prior, rules).allowed is True


def test_both_limits_reported_together(rules: PolicyRules) -> None:
    prior = [jst(2026, 10, 20, 10), jst(2026, 10, 22, 10), jst(2026, 10, 24, 9)]
    d = check_attempt_limits(NOW, "Asia/Tokyo", prior, rules)
    assert d.codes == ("daily_limit_reached", "weekly_limit_reached")


def test_local_day_uses_debtor_timezone(rules: PolicyRules) -> None:
    # Prior: 16:00 UTC Oct 23 = 01:00 JST Oct 24 = 12:00 EDT Oct 23.
    # Now:   05:00 UTC Oct 24 = 14:00 JST Oct 24 = 01:00 EDT Oct 24.
    # Same local day in Tokyo, different local day in New York.
    prior = [datetime(2026, 10, 23, 16, tzinfo=UTC)]
    now = datetime(2026, 10, 24, 5, tzinfo=UTC)
    assert check_attempt_limits(now, "Asia/Tokyo", prior, rules).allowed is False
    assert check_attempt_limits(now, "America/New_York", prior, rules).allowed is True


def test_attempt_limits_invalid_timezone_fails_closed(rules: PolicyRules) -> None:
    d = check_attempt_limits(NOW, "Not/AZone", [], rules)
    assert d.allowed is False
    assert d.codes == ("invalid_timezone",)


def test_attempt_limits_reject_naive_attempt(rules: PolicyRules) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        check_attempt_limits(NOW, "Asia/Tokyo", [datetime(2026, 10, 24, 9)], rules)


# --- Stop-contact -----------------------------------------------------------------------


def test_stop_contact_blocks() -> None:
    d = check_stop_contact(NOW, stop_contact=True)
    assert d.allowed is False
    assert d.codes == ("stop_contact_requested",)


def test_no_stop_contact_allows() -> None:
    assert check_stop_contact(NOW, stop_contact=False).allowed is True


# --- Workplace contact ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "flag", "justification", "allowed", "code"),
    [
        (DialTarget.MOBILE, False, None, True, None),
        (DialTarget.HOME, False, None, True, None),
        (DialTarget.MOBILE, True, None, True, None),
        (DialTarget.WORK, False, "no other number reachable for 30 days", True, None),
        (DialTarget.WORK, False, None, False, "no_workplace_justification"),
        (DialTarget.WORK, False, "   ", False, "no_workplace_justification"),
        (DialTarget.WORK, True, "manager approved", False, "debtor_refused_workplace_contact"),
        (DialTarget.WORK, True, None, False, "debtor_refused_workplace_contact"),
    ],
)
def test_workplace_contact(
    target: DialTarget, flag: bool, justification: str | None, allowed: bool, code: str | None
) -> None:
    d = check_workplace_contact(NOW, target, flag, justification)
    assert d.allowed is allowed
    assert d.codes == ((code,) if code else ())


def test_workplace_justification_text_not_logged() -> None:
    d = check_workplace_contact(NOW, DialTarget.WORK, False, "debtor's employer is ACME")
    assert "ACME" not in str(d.input)


# --- Pre-call check ---------------------------------------------------------------------


def _pre_call(rules: PolicyRules, **overrides: object) -> list[bool]:
    args: dict[str, object] = {
        "debtor_tz": "Asia/Tokyo",
        "prior_attempts": [],
        "stop_contact": False,
        "target": DialTarget.MOBILE,
        "no_workplace_contact": False,
        "workplace_justification": None,
        "rules": rules,
    } | overrides
    decisions = pre_call_check(args.pop("now", NOW), **args)  # type: ignore[arg-type]
    return [d.allowed for d in decisions]


def test_pre_call_all_clear(rules: PolicyRules) -> None:
    assert _pre_call(rules) == [True, True, True, True]


def test_pre_call_returns_every_rule(rules: PolicyRules) -> None:
    decisions = pre_call_check(
        NOW,
        debtor_tz="Asia/Tokyo",
        prior_attempts=[],
        stop_contact=False,
        target=DialTarget.MOBILE,
        no_workplace_contact=False,
        workplace_justification=None,
        rules=rules,
    )
    assert [d.rule for d in decisions] == [
        Rule.STOP_CONTACT,
        Rule.CONTACT_WINDOW,
        Rule.ATTEMPT_LIMITS,
        Rule.WORKPLACE_CONTACT,
    ]


@pytest.mark.parametrize(
    ("overrides", "blocked_index"),
    [
        ({"stop_contact": True}, 0),
        ({"now": jst(2026, 10, 24, 22)}, 1),
        ({"prior_attempts": [jst(2026, 10, 24, 9)]}, 2),
        ({"target": DialTarget.WORK}, 3),
    ],
)
def test_pre_call_single_failure_blocks(
    rules: PolicyRules, overrides: dict[str, object], blocked_index: int
) -> None:
    results = _pre_call(rules, **overrides)
    assert not all(results)
    assert results[blocked_index] is False
    assert sum(1 for r in results if not r) == 1
