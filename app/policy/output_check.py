"""Output interceptor (pure). Runs on every LLM-written agent reply before it is sent.

Per sentence, in order: banned language (always) -> before verification, any debt word or
figure -> after verification, every figure must be an allowed figure. On the first
violating sentence: drop it and everything after it, append the safe fallback.

Allowed figures come only from code (balance, history, approved terms, rule limits).
Figures the debtor offered are not allowed until propose_terms approves them.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from app.policy.config import PolicyRules
from app.policy.engine import Decision, Rule

MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for m in names
}
# Also ordinary words: only a date when a day number is next to them.
_AMBIGUOUS_MONTHS = {"may", "march", "mar", "jan", "jun", "sep", "dec", "oct", "aug", "apr"}

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}  # fmt: skip
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}
_NUMBER_WORDS = {*_ONES, *_TENS, *_SCALES}
_FRACTIONS = {
    "half": 50, "one half": 50, "a third": 33, "one third": 33, "two thirds": 67,
    "a quarter": 25, "one quarter": 25, "three quarters": 75,
}  # fmt: skip

_COUNT_UNITS = r"(?:months?|installments?|instalments?|payments?|times)"
# v4 (monthly_counts): "six monthly payments" is the count 6, like "six payments".
_MONTHLY_BEFORE_UNIT = re.compile(
    r"\bmonthly\s+(?=(?:payments?|installments?|instalments?)\b)", re.I
)
# Shared with the voice sink, so voice splits a reply exactly where this check does.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?。])\s+")
_ISO_DATE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
_MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))
# v2 (full_dates): "March 1, 2027" / "1 March 2027" is one date, not a date plus amount 2027.
_MONTH_DAY_YEAR = re.compile(
    rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I
)
_DAY_MONTH_YEAR = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALT}),?\s+(\d{{4}})\b", re.I
)
# v4 (monthly_counts): "in May 2027" is a month of a year, not the month May plus amount 2027.
_MONTH_YEAR = re.compile(rf"\b({_MONTH_ALT})\.?,?\s+(\d{{4}})\b", re.I)
_MONTH_DAY = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.I)
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALT})\b", re.I)
_MONTH_ALONE = re.compile(rf"\b({_MONTH_ALT})\b", re.I)
_AMOUNT = re.compile(r"(?:¥|\bJPY\s?)\s?(\d[\d,]*)|\b(\d[\d,]*)\s?(?:yen|JPY)\b", re.I)
_PERCENT = re.compile(r"\b(\d+(?:\.\d+)?)\s?(?:%|percent\b|per cent\b)", re.I)
_BARE_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*", re.I)


@dataclass(frozen=True)
class AllowedFigures:
    amounts: frozenset[int] = frozenset()
    counts: frozenset[int] = frozenset()
    percents: frozenset[int] = frozenset()
    dates: frozenset[date] = frozenset()


@dataclass(frozen=True)
class Figure:
    kind: str  # amount | count | percent | date | month_day | month | relative_date
    value: Any

    def label(self) -> str:
        return f"{self.kind}:{self.value}"


@dataclass(frozen=True)
class _Violation:
    code: str
    detail: str


@dataclass(frozen=True)
class OutputCheckResult:
    text: str
    decision: Decision
    dropped_sentences: int = field(default=0)


def _phrase_regex(phrases: tuple[str, ...]) -> re.Pattern[str]:
    alt = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
    return re.compile(rf"\b(?:{alt})\b", re.I)


def _blank(text: str, span: tuple[int, int]) -> str:
    return text[: span[0]] + " " * (span[1] - span[0]) + text[span[1] :]


_QUALIFIER = re.compile(r"\b(next|this|last|coming)\s+$", re.I)


def relative_phrases(text: str, relative_dates: re.Pattern[str]) -> set[str]:
    """Relative-date phrases with their qualifier, lowercased: 'friday', 'next friday',
    'next month'. 'Friday' and 'next Friday' are different phrases."""
    norm = " ".join(unicodedata.normalize("NFKC", text).split())
    found: set[str] = set()
    for m in relative_dates.finditer(norm):
        phrase = m.group(0).lower()
        q = _QUALIFIER.search(norm[: m.start()])
        found.add(f"{q[1].lower()} {phrase}" if q else phrase)
    return found


def customer_relative_phrases(said: list[str], rules: PolicyRules) -> frozenset[str]:
    """Agent v4 (fixes.echo_relative_dates): relative dates the person said in this call."""
    pattern = _phrase_regex(rules.output_check.relative_date_words)
    return frozenset(p for line in said for p in relative_phrases(line, pattern))


def customer_month_years(said: list[str]) -> frozenset[tuple[int, int]]:
    """Agent v5 (fixes.echo_month_year, V5-5): (year, month) pairs the person said in this
    call, like 'May 2027'. The agent may repeat them; no other month-year becomes allowed."""
    return frozenset(
        (int(m[2]), MONTHS[m[1].lower()])
        for line in said
        for m in _MONTH_YEAR.finditer(unicodedata.normalize("NFKC", line))
    )


def _int(raw: str) -> int:
    return int(float(raw.replace(",", "")))


def _words_to_number(words: list[str]) -> int:
    total, current = 0, 0
    for w in words:
        if w in _ONES:
            current += _ONES[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = max(current, 1) * 100
        elif w in ("thousand", "million"):
            total += max(current, 1) * _SCALES[w]
            current = 0
    return total + current


def _number_word_figures(text: str) -> list[Figure]:
    """'three months' -> count 3, 'thirty thousand yen' -> amount 30000, 'ten percent' -> 10%.
    Bare small number words ('one moment') are ignored to avoid false positives."""
    tokens = [m.group(0).lower() for m in _WORD.finditer(text.replace("-", " "))]
    figures: list[Figure] = []
    i = 0
    while i < len(tokens):
        if tokens[i] not in _NUMBER_WORDS and not (
            tokens[i] == "a" and i + 1 < len(tokens) and tokens[i + 1] in _SCALES
        ):
            i += 1
            continue
        run: list[str] = []
        while i < len(tokens) and (tokens[i] in _NUMBER_WORDS or tokens[i] in ("and", "a")):
            if tokens[i] in _NUMBER_WORDS:
                run.append(tokens[i])
            i += 1
        if not run:
            continue
        value = _words_to_number(run)
        unit = tokens[i] if i < len(tokens) else ""
        if unit == "yen":
            figures.append(Figure("amount", value))
        elif unit in ("percent", "per"):
            figures.append(Figure("percent", value))
        elif re.fullmatch(_COUNT_UNITS, unit):
            figures.append(Figure("count", value))
        elif value >= 1_000:
            figures.append(Figure("amount", value))
    return figures


def _date_figure(year: str, month: int, day: str, raw: str) -> Figure:
    try:
        return Figure("date", date(int(year), month, int(day)))
    except ValueError:
        return Figure("date", raw)  # not a real date: never an allowed one


def extract_figures(
    sentence: str,
    relative_dates: re.Pattern[str],
    *,
    full_dates: bool = False,
    monthly_counts: bool = False,
) -> list[Figure]:
    """Every money amount, count, percentage and date mentioned in one sentence.

    full_dates (agent v2): a month-day with its year is read as one full date.
    monthly_counts (agent v4): "six monthly payments" is read as the count 6, and "May 2027"
    as a month of a year (allowed when an approved date falls in it)."""
    text = _MONTHLY_BEFORE_UNIT.sub("", sentence) if monthly_counts else sentence
    figures: list[Figure] = []

    for m in list(_ISO_DATE.finditer(text)):
        figures.append(_date_figure(m[1], int(m[2]), m[3], m.group(0)))
        text = _blank(text, m.span())
    if full_dates:
        for m in list(_MONTH_DAY_YEAR.finditer(text)):
            figures.append(_date_figure(m[3], MONTHS[m[1].lower()], m[2], m.group(0)))
            text = _blank(text, m.span())
        for m in list(_DAY_MONTH_YEAR.finditer(text)):
            figures.append(_date_figure(m[3], MONTHS[m[2].lower()], m[1], m.group(0)))
            text = _blank(text, m.span())
    if monthly_counts:
        for m in list(_MONTH_YEAR.finditer(text)):
            figures.append(Figure("month_year", (int(m[2]), MONTHS[m[1].lower()])))
            text = _blank(text, m.span())
    for pattern, month_group, day_group in ((_MONTH_DAY, 1, 2), (_DAY_MONTH, 2, 1)):
        for m in list(pattern.finditer(text)):
            figures.append(Figure("month_day", (MONTHS[m[month_group].lower()], int(m[day_group]))))
            text = _blank(text, m.span())
    for m in list(_MONTH_ALONE.finditer(text)):
        if m[1].lower() not in _AMBIGUOUS_MONTHS:
            figures.append(Figure("month", MONTHS[m[1].lower()]))
            text = _blank(text, m.span())
    for m in list(relative_dates.finditer(text)):
        figures.append(Figure("relative_date", m.group(0).lower()))
        text = _blank(text, m.span())

    lowered = text.lower()
    for phrase, pct in sorted(_FRACTIONS.items(), key=lambda kv: -len(kv[0])):
        for m in list(re.finditer(rf"\b{re.escape(phrase)}\b", lowered)):
            figures.append(Figure("percent", pct))
            lowered = _blank(lowered, m.span())
            text = _blank(text, m.span())

    for m in list(_AMOUNT.finditer(text)):
        figures.append(Figure("amount", _int(m[1] or m[2])))
        text = _blank(text, m.span())
    for m in list(_PERCENT.finditer(text)):
        figures.append(Figure("percent", _int(m[1])))
        text = _blank(text, m.span())
    for m in list(_BARE_NUMBER.finditer(text)):
        value = _int(m.group(0))
        figures.append(Figure("amount" if value >= 1_000 else "count", value))
        text = _blank(text, m.span())

    figures += _number_word_figures(text)
    return figures


def _allowed(fig: Figure, allowed: AllowedFigures) -> bool:
    match fig.kind:
        case "amount":
            return bool(fig.value in allowed.amounts)
        case "count":
            return bool(fig.value in allowed.counts)
        case "percent":
            return bool(fig.value in allowed.percents)
        case "date":
            return bool(fig.value in allowed.dates)
        case "month_day":
            return any((d.month, d.day) == fig.value for d in allowed.dates)
        case "month":
            return any(d.month == fig.value for d in allowed.dates)
        case "month_year":
            return any((d.year, d.month) == fig.value for d in allowed.dates)
        case _:  # relative_date: can never be matched to an approved date
            return False


def _first_violation(
    sentence: str,
    *,
    identity_verified: bool,
    allowed: AllowedFigures,
    banned: re.Pattern[str],
    debt_words: re.Pattern[str],
    relative_dates: re.Pattern[str],
    full_dates: bool,
    monthly_counts: bool,
    customer_relative: frozenset[str] | None,
    unverified: re.Pattern[str] | None,
    customer_months: frozenset[tuple[int, int]] = frozenset(),
) -> _Violation | None:
    if m := banned.search(sentence):
        return _Violation("banned_language", m.group(0).lower())
    if unverified is not None and (m := unverified.search(sentence)):
        return _Violation("unverified_claim", m.group(0).lower())
    figures = extract_figures(
        sentence, relative_dates, full_dates=full_dates, monthly_counts=monthly_counts
    )
    if not identity_verified:
        if m := debt_words.search(sentence):
            return _Violation("pre_verification_disclosure", f"word:{m.group(0).lower()}")
        if figures:
            return _Violation("pre_verification_disclosure", figures[0].label())
        return None
    echo = customer_relative is not None
    for fig in figures:
        if echo and fig.kind == "relative_date":
            continue  # checked as whole phrases below
        if fig.kind == "month_year" and fig.value in customer_months:
            continue  # v5 echo_month_year: the person's own month and year
        if not _allowed(fig, allowed):
            code = "relative_date" if fig.kind == "relative_date" else "unapproved_figure"
            return _Violation(code, fig.label())
    if customer_relative is not None:
        # v4 echo: a relative date is allowed only in the person's own words from this call.
        unsaid = sorted(relative_phrases(sentence, relative_dates) - customer_relative)
        if unsaid:
            return _Violation("relative_date", f"relative_date:{unsaid[0]}")
    return None


def check_output(
    now: datetime,
    text: str,
    *,
    identity_verified: bool,
    allowed: AllowedFigures,
    rules: PolicyRules,
    fallback: str,
    full_dates: bool = False,
    monthly_counts: bool = False,
    customer_relative: frozenset[str] | None = None,
    claims_check: bool = False,
    customer_months: frozenset[tuple[int, int]] | None = None,
) -> OutputCheckResult:
    """One decision per reply. Blocked: keep sentences before the violation, add fallback.

    full_dates: the agent version's fixes.full_date_parsing (False = v1 behavior).
    monthly_counts: fixes.llm_presents_terms (agent v4); off for v1-v3.
    customer_relative: fixes.echo_relative_dates (agent v4): relative dates the person said in
    this call, which the agent may repeat word for word. None = off (never allowed).
    claims_check: fixes.no_interest_claims (agent v5): rules.claims phrases are never said.
    customer_months: fixes.echo_month_year (agent v5): (year, month) pairs the person said,
    which the agent may repeat. None = off."""
    lists = rules.output_check
    sentences = [s for s in SENTENCE_SPLIT.split(text.strip()) if s]
    checks = ["banned_language", "pre_verification" if not identity_verified else "figures"]
    inp: dict[str, Any] = {
        "identity_verified": identity_verified,
        "checks": checks,
        "sentences": len(sentences),
        "rules_version": rules.version,
        "full_dates": full_dates,
    }
    if monthly_counts:  # only v4 rows carry the key: v1-v3 events stay as they were
        inp["monthly_counts"] = True
    if customer_relative is not None:
        inp["customer_relative_dates"] = sorted(customer_relative)
    if customer_months is not None:
        inp["customer_month_years"] = sorted(f"{y}-{m:02d}" for y, m in customer_months)
    unverified = None
    if claims_check:
        if rules.claims is None:
            raise ValueError(f"{rules.version} has no claims block (rules-v8 or later)")
        unverified = _phrase_regex(rules.claims.unverified_phrases)
        inp["claims_check"] = True
    patterns = {
        "banned": _phrase_regex(lists.banned_phrases),
        "debt_words": _phrase_regex(lists.debt_words),
        "relative_dates": _phrase_regex(lists.relative_date_words),
    }
    for index, sentence in enumerate(sentences):
        violation = _first_violation(
            sentence,
            identity_verified=identity_verified,
            allowed=allowed,
            full_dates=full_dates,
            monthly_counts=monthly_counts,
            customer_relative=customer_relative,
            unverified=unverified,
            customer_months=customer_months or frozenset(),
            **patterns,
        )
        if violation is None:
            continue
        kept = sentences[:index]
        dropped = len(sentences) - index
        inp |= {
            "kept": index,
            "dropped": dropped,
            "violation": violation.code,
            "detail": violation.detail,
            "offending_sentence": sentence[:300],
        }
        decision = Decision(
            rule=Rule.OUTPUT_CHECK,
            allowed=False,
            reason=f"{violation.code} ({violation.detail}) in sentence {index + 1}",
            at=now,
            input=inp,
            codes=(violation.code,),
        )
        return OutputCheckResult(" ".join([*kept, fallback]), decision, dropped)
    inp |= {"kept": len(sentences), "dropped": 0}
    decision = Decision(
        rule=Rule.OUTPUT_CHECK, allowed=True, reason="output passed checks", at=now, input=inp
    )
    return OutputCheckResult(text.strip(), decision)


def banned_phrase(text: str, rules: PolicyRules) -> str | None:
    """The first banned phrase in the text, if any (used by eval checks on delivered turns)."""
    m = _phrase_regex(rules.output_check.banned_phrases).search(text)
    return m.group(0).lower() if m else None


def check_terms_presented(
    now: datetime,
    *,
    terms_id: int,
    required: list[Figure],
    said: list[str],
    rules: PolicyRules,
) -> Decision:
    """Agent v4 (fixes.llm_presents_terms): the LLM says approved terms in its own words, so a
    promise may be recorded only once every required figure was actually said to the person
    (down payment, monthly amount and count; or the full amount and its date).

    `said`: the guarded agent text delivered since the terms were approved."""
    relative = _phrase_regex(rules.output_check.relative_date_words)
    heard = [
        f
        for line in said
        for f in extract_figures(line, relative, full_dates=True, monthly_counts=True)
    ]

    def _heard(fig: Figure) -> bool:
        if fig.kind == "date":
            return any(
                h.value == fig.value
                or (h.kind == "month_day" and h.value == (fig.value.month, fig.value.day))
                for h in heard
            )
        return any(h.kind == fig.kind and h.value == fig.value for h in heard)

    missing = [f.label() for f in required if not _heard(f)]
    inp: dict[str, Any] = {
        "terms_id": terms_id,
        "required": [f.label() for f in required],
        "missing": missing,
        "rules_version": rules.version,
    }
    if missing:
        return Decision(
            rule=Rule.TERMS_PRESENTED,
            allowed=False,
            reason=f"terms not fully said to the person: missing {', '.join(missing)}",
            at=now,
            input=inp,
            codes=("terms_not_presented",),
        )
    return Decision(
        rule=Rule.TERMS_PRESENTED,
        allowed=True,
        reason="all approved terms were said",
        at=now,
        input=inp,
    )
