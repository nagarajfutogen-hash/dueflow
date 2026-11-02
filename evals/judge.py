"""LLM judge (Groq, a different family from the agent and simulator).

Scores tone, empathy, clarity and de-escalation with a reason each. It may also report
language violations, each with an exact quote. A quote is trusted only if code finds it in a
delivered LLM-written agent turn. Fixed lines rendered by code (opening, verification, terms,
goodbyes) are policy-approved and unit-tested, so they are never judge violations (rubric-v1
flagged the required opening: eval run #4). Untrusted quotes are kept as evidence only.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.llm.groq_chat import ChatResult, GroqChat, InvalidJSONError, LLMUnavailableError

RUBRIC_PATH = Path(__file__).resolve().parent / "rubric.yaml"
MIN_QUOTE_CHARS = 8
# Invalid JSON (ours to parse, or Groq's json_validate_failed): at most 2 retries, each
# with a new seed.
INVALID_JSON_RETRIES = 2
RETRY_SEED_OFFSET = 7_919
# 900 cut the 4-criterion reply short (run #20: every call used exactly 900 completion
# tokens and lost the last score, de_escalation). Room only; the rubric is unchanged.
JUDGE_MAX_TOKENS = 1500
_SPACES = re.compile(r"\s+")


class Rubric(BaseModel):
    model_config = ConfigDict(frozen=True)
    version: str
    scale: str
    criteria: dict[str, str]
    violation_types: dict[str, str]


def load_rubric(path: Path = RUBRIC_PATH) -> Rubric:
    return Rubric.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class _Score(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str = Field(min_length=1)


class _Violation(BaseModel):
    type: str
    quote: str


class _JudgeOut(BaseModel):
    scores: dict[str, _Score]
    violations: list[_Violation] = []


@dataclass
class JudgeResult:
    scores: dict[str, tuple[int, str]] = field(default_factory=dict)
    violations: list[dict[str, str]] = field(default_factory=list)  # quote found in transcript
    unverified: list[dict[str, str]] = field(default_factory=list)  # quote not found: ignored
    usage: list[ChatResult] = field(default_factory=list)
    error: str | None = None


def _norm(text: str) -> str:
    return _SPACES.sub(" ", text).strip().casefold()


def transcript_text(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{'AGENT' if who == 'agent' else 'PERSON'}: {text}" for who, text in turns)


def build_messages(rubric: Rubric, turns: list[tuple[str, str]]) -> list[dict[str, str]]:
    criteria = "\n".join(f"- {name}: {desc}" for name, desc in rubric.criteria.items())
    kinds = "\n".join(f"- {name}: {desc}" for name, desc in rubric.violation_types.items())
    schema = {
        "scores": {name: {"score": "1-5", "reason": "one sentence"} for name in rubric.criteria},
        "violations": [{"type": "one of the violation types", "quote": "exact AGENT words"}],
    }
    system = (
        "You grade the AGENT's language in a debt-collection call transcript. Grade only how "
        "the AGENT speaks, never the PERSON. Policy compliance is checked elsewhere.\n"
        f"Scale: {rubric.scale}\nCriteria:\n{criteria}\n"
        f"Violation types (report only clear cases, quoting the AGENT exactly):\n{kinds}\n"
        f"Reply with JSON only, shaped like: {json.dumps(schema)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": transcript_text(turns)},
    ]


def parse(raw: str, rubric: Rubric, agent_lines: list[str]) -> JudgeResult:
    out = _JudgeOut.model_validate(json.loads(raw))
    missing = set(rubric.criteria) - set(out.scores)
    if missing:
        raise ValueError(f"missing scores: {sorted(missing)}")
    result = JudgeResult(scores={k: (v.score, v.reason) for k, v in out.scores.items()
                                 if k in rubric.criteria})  # fmt: skip
    haystack = [_norm(line) for line in agent_lines]
    for v in out.violations:
        quote = _norm(v.quote)
        item = {"type": v.type, "quote": v.quote[:300]}
        known = v.type in rubric.violation_types
        if known and len(quote) >= MIN_QUOTE_CHARS and any(quote in line for line in haystack):
            result.violations.append(item)
        else:
            result.unverified.append(item)
    return result


async def judge(
    chat: GroqChat,
    model: str,
    rubric: Rubric,
    turns: list[tuple[str, str]],
    seed: int,
    llm_lines: list[str] | None = None,
) -> JudgeResult:
    """`turns` is the whole delivered conversation (the judge sees context); `llm_lines` are
    the agent lines the LLM wrote, the only ones a violation may be attributed to."""
    agent_lines = llm_lines if llm_lines is not None else [t for w, t in turns if w == "agent"]
    messages = build_messages(rubric, turns)
    usage: list[ChatResult] = []
    last_error = ""
    for attempt in range(1 + INVALID_JSON_RETRIES):
        try:
            reply = await chat.complete(
                model=model,
                messages=messages,
                temperature=0.0,
                # temperature 0 + the same seed repeats the same output: re-sample with another
                seed=seed + attempt * RETRY_SEED_OFFSET,
                max_tokens=JUDGE_MAX_TOKENS,
                json_mode=True,
            )
        except InvalidJSONError as exc:  # Groq rejected its own output as invalid JSON
            last_error = f"invalid json: {exc}"
            continue
        except LLMUnavailableError as exc:
            return JudgeResult(usage=usage, error=f"unavailable: {exc}")
        usage.append(reply)
        try:
            result = parse(reply.text, rubric, agent_lines)
        except (ValueError, ValidationError) as exc:
            last_error = f"invalid output: {str(exc)[:200]}"
            continue
        result.usage = usage
        return result
    return JudgeResult(usage=usage, error=last_error)


def as_evidence(result: JudgeResult) -> dict[str, Any]:
    return {"violations": result.violations, "unverified": result.unverified}
