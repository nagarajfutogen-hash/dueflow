"""Pure phrase matching on the person's own words (stdlib only, no I/O).

Used by rules-v7 claims: "this is a wrong number" and "I already paid". Matching is
case-, width- and apostrophe-insensitive on word boundaries; "do not" / "dont" read as "don't".
"""

import re
import unicodedata
from collections.abc import Iterable

from app.policy.identity import normalize_name

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'"})
_SPACES = re.compile(r"\s+")
_DONT = re.compile(r"\b(?:do not|dont)\b")
NAME_SLOT = "{name}"


def normalize_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw).casefold().translate(_APOSTROPHES)
    text = _SPACES.sub(" ", text).strip()
    return _DONT.sub("don't", text)


def name_variants(full_name: str) -> tuple[str, ...]:
    """The full name plus each name token ("Haru Wrongtest" -> full, "haru", "wrongtest")."""
    tokens = normalize_name(full_name)
    full = " ".join(normalize_text(full_name).split())
    return tuple(dict.fromkeys([full, *tokens]))


def find_phrase(text: str, phrases: Iterable[str], names: Iterable[str] = ()) -> str | None:
    """The first listed phrase (as written in the rules, {name} unfilled) found in the text."""
    said = normalize_text(text)
    names = tuple(names)
    for phrase in phrases:
        template = normalize_text(phrase)
        variants = (
            [template.replace(NAME_SLOT, n) for n in names] if NAME_SLOT in template else [template]
        )
        for v in variants:
            if re.search(rf"(?<![\w']){re.escape(v)}(?![\w'])", said):
                return phrase
    return None
