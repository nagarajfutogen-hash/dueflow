"""Pure normalization for identity checks (stdlib only, no I/O)."""

import re
import unicodedata
from datetime import date

_PUNCT = re.compile(r"[.,'’\-_/()·・]")
_SPACES = re.compile(r"\s+")

_MONTHS = {
    name: i
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
    for name in names
}


def normalize_name(raw: str) -> tuple[str, ...]:
    """Case/width/spacing/punctuation-insensitive, order-insensitive name tokens."""
    text = unicodedata.normalize("NFKC", raw).casefold()
    text = _PUNCT.sub(" ", text)
    return tuple(sorted(t for t in _SPACES.split(text.strip()) if t))


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


_FIND_YMD = re.compile(r"(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})\s*日?")
_FIND_COMPACT = re.compile(r"\b(\d{4})(\d{2})(\d{2})\b")
# V5-7 voice: "," or "." between the parts (live call 4: one date split into two turns,
# "April twelfth." + "1985.", read joined).
_FIND_MDY = re.compile(r"\b([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?[,.]?\s+(\d{4})\b")
_FIND_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]+)[,.]?\s+(\d{4})\b")


_UNIT_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_TEEN_WORDS = (
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
)  # fmt: skip
_TENS_WORDS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_UNIT_ORD = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth")
_TEEN_ORD = (
    "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
    "seventeenth", "eighteenth", "nineteenth",
)  # fmt: skip


def _number_words() -> tuple[dict[str, int], dict[str, int]]:
    """Spoken 1-99 (cardinal) and 1-31 (cardinal or ordinal, as a day of the month)."""
    upto99: dict[str, int] = {w: i for i, w in enumerate(_UNIT_WORDS, 1)}
    upto99 |= {w: i for i, w in enumerate(_TEEN_WORDS, 10)}
    days: dict[str, int] = {w: i for i, w in enumerate(_UNIT_ORD, 1)}
    days |= {w: i for i, w in enumerate(_TEEN_ORD, 10)}
    days |= {"twentieth": 20, "thirtieth": 30}
    for t, tens in enumerate(_TENS_WORDS, 2):
        upto99[tens] = t * 10
        for u, unit in enumerate(_UNIT_WORDS, 1):
            upto99[f"{tens} {unit}"] = t * 10 + u
            if t * 10 + u <= 31:
                days[f"{tens} {_UNIT_ORD[u - 1]}"] = t * 10 + u
    days |= {w: n for w, n in upto99.items() if n <= 31}
    return upto99, days


_UPTO99, _DAY_WORDS = _number_words()
# Birth years as people say them: "nineteen eighty four", "nineteen oh five", "nineteen
# hundred", "two thousand", "two thousand and one". ("twenty nineteen" is left out: it
# collides with "April twenty, nineteen ninety".)
_YEAR_WORDS: dict[str, int] = {"nineteen hundred": 1900, "two thousand": 2000}
_YEAR_WORDS |= {f"nineteen {w}": 1900 + n for w, n in _UPTO99.items() if n >= 10}
_YEAR_WORDS |= {f"nineteen oh {w}": 1900 + n for w, n in _UPTO99.items() if n < 10}
_YEAR_WORDS |= {f"two thousand {w}": 2000 + n for w, n in _UPTO99.items()}
_YEAR_WORDS |= {f"two thousand and {w}": 2000 + n for w, n in _UPTO99.items()}


def _alternation(words: dict[str, int]) -> re.Pattern[str]:
    alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b(?:{alt})\b")


_YEAR_RE = _alternation(_YEAR_WORDS)
_DAY_RE = _alternation(_DAY_WORDS)


def spoken_numbers_to_digits(text: str) -> str:
    """V5-7 voice: speech-to-text can leave a date in words ("April twentieth nineteen eighty
    four"); years first, then days, become digits so the date patterns below read them.
    Lowercase input. Only number words change; everything else stays as it was."""
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)  # "eighty-four" -> "eighty four"
    text = _YEAR_RE.sub(lambda m: str(_YEAR_WORDS[m.group(0)]), text)
    return _DAY_RE.sub(lambda m: str(_DAY_WORDS[m.group(0)]), text)


def find_dates(raw: str) -> set[date]:
    """Every unambiguous date written anywhere in free text (day/month-ambiguous forms skipped).
    Dates said in words are read too (voice, V5-7)."""
    text = spoken_numbers_to_digits(unicodedata.normalize("NFKC", raw).casefold())
    found: set[date] = set()
    for m in (*_FIND_YMD.finditer(text), *_FIND_COMPACT.finditer(text)):
        if d := _safe_date(int(m[1]), int(m[2]), int(m[3])):
            found.add(d)
    for m in _FIND_MDY.finditer(text):
        if m[1] in _MONTHS and (d := _safe_date(int(m[3]), _MONTHS[m[1]], int(m[2]))):
            found.add(d)
    for m in _FIND_DMY.finditer(text):
        if m[2] in _MONTHS and (d := _safe_date(int(m[3]), _MONTHS[m[2]], int(m[1]))):
            found.add(d)
    return found


def name_in_text(full_name: str, raw: str) -> bool:
    """All tokens of the name appear in the text (order-insensitive, normalized)."""
    wanted = set(normalize_name(full_name))
    return bool(wanted) and wanted <= set(normalize_name(raw))
