from datetime import UTC, date, datetime

from app.policy.engine import PaymentClaim, PaymentOnFile, Rule, check_payment_record

NOW = datetime(2026, 10, 30, 3, tzinfo=UTC)
ON_FILE = [
    PaymentOnFile(40_000, date(2026, 9, 5), "card", "CARD-7781"),
    PaymentOnFile(20_000, date(2026, 10, 5), "convenience_store", "CVS-55120"),
]


def _check(**claim: object) -> tuple[bool, str | None, tuple[str, ...]]:
    d = check_payment_record(NOW, claim=PaymentClaim(**claim), payments=ON_FILE)  # type: ignore[arg-type]
    assert d.rule is Rule.PAYMENT_RECORD
    return d.allowed, d.input.get("match"), d.codes


def test_reference_matches_ignoring_case_and_punctuation() -> None:
    assert _check(reference="cvs 55120") == (True, "reference", ())


def test_amount_and_date_match() -> None:
    assert _check(amount_jpy=40_000, paid_on=date(2026, 9, 5)) == (True, "amount_and_date", ())


def test_same_amount_on_another_date_is_a_possible_match() -> None:
    assert _check(amount_jpy=20_000, paid_on=date(2026, 10, 1)) == (True, "amount_only", ())


def test_no_matching_payment_is_blocked() -> None:
    allowed, _, codes = _check(amount_jpy=60_000, paid_on=date(2026, 10, 20), reference="X-1")
    assert not allowed and codes == ("no_matching_payment",)


def test_no_details_is_blocked() -> None:
    assert _check(method="card") == (False, None, ("claim_details_missing",))


def test_nothing_on_file_never_matches() -> None:
    d = check_payment_record(NOW, claim=PaymentClaim(amount_jpy=1), payments=[])
    assert not d.allowed and d.codes == ("no_matching_payment",)
