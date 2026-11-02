"""Day 3 eval machinery, tested without any LLM call."""

import asyncio
import json
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from app.agent.conversation import OpenConversation
from app.agent.scripts import VERIFIED, opening
from app.db.models import EvalSuite, Speaker
from app.llm.groq_chat import ChatResult, InvalidJSONError, LLMUnavailableError
from app.policy.config import load_rules
from app.policy.output_check import AllowedFigures
from evals import runner
from evals.checks import _Data, find_critical
from evals.judge import JUDGE_MAX_TOKENS, judge, load_rubric, parse
from evals.report import render
from evals.run import scenario_seed
from evals.scenario import SUITE_FILES, load_suite
from evals.simulate import HANGUP, Turn, next_line, parse_line, system_prompt, to_messages

RULES = load_rules()
RUBRIC = load_rubric()
NAME = "Aoi Checktest"
T0 = datetime(2026, 10, 26, 3, tzinfo=UTC)
NOW = T0 + timedelta(minutes=5)


def _scenario(suite: EvalSuite, scenario_id: str) -> Any:
    return next(s for s in load_suite(SUITE_FILES[suite]) if s.id == scenario_id)


# --- Simulator ------------------------------------------------------------------------------


def test_debtor_prompt_knows_name_and_dob() -> None:
    s = _scenario(EvalSuite.BASE, "base/cooperative")
    prompt = system_prompt(s)
    assert s.debtor.full_name in prompt and s.debtor.dob.isoformat() in prompt
    assert HANGUP in prompt


def test_third_party_and_stranger_do_not_know_the_dob() -> None:
    for suite, sid in [
        (EvalSuite.ADAPTIVE, "adaptive/unverified_extraction"),
        (EvalSuite.BASE, "base/wrong_person"),
    ]:
        s = _scenario(suite, sid)
        assert s.debtor.dob.isoformat() not in system_prompt(s)
        assert "You are NOT" in system_prompt(s)


def test_adaptive_prompt_is_a_red_team_role() -> None:
    assert "red-team" in system_prompt(_scenario(EvalSuite.ADAPTIVE, "adaptive/discount_hunter"))


def test_messages_from_the_debtor_side() -> None:
    s = _scenario(EvalSuite.BASE, "base/cooperative")
    msgs = to_messages(s, [Turn("agent", "Hello?"), Turn("debtor", "Hi.")])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]


@pytest.mark.parametrize(
    ("raw", "text", "hang_up"),
    [
        (" Sure, that works. ", "Sure, that works.", False),
        ('"Okay."', "Okay.", False),
        (HANGUP, None, True),
        ("", None, False),
        # sim-v1 defect: these words were dropped, so the agent never heard the acceptance.
        ("That works for me.[HANGUP]", "That works for me.", True),
        ("Please stop calling me. [HANGUP]", "Please stop calling me.", True),
    ],
)
def test_parse_line(raw: str, text: str | None, hang_up: bool) -> None:
    line = parse_line(raw)
    assert (line.text, line.hang_up, line.raw) == (text, hang_up, raw)


@pytest.mark.parametrize(
    "raw",
    [
        # sim-v2 faults seen in runs #10 and #16-#18 (base/angry, base/hardship).
        "Yeah, that's me.[HANGUP]",
        "Yeah, it's me.[HANGUP]",
        "…until I get back on my feet?[HANGUP]",
        "Yes, this is Aoi Checktest. [HANGUP]",
        "Yes, speaking.[HANGUP]",
        "Yeah, that’s me. [HANGUP]",
    ],
)
def test_incoherent_hang_up_is_dropped(raw: str) -> None:
    line = parse_line(raw, NAME, allow_drop=True)
    assert line.text and not line.hang_up and line.hang_up_dropped and line.raw == raw


@pytest.mark.parametrize(
    "raw",
    [
        "That works for me.[HANGUP]",  # goal reached
        "Please stop calling me. [HANGUP]",  # clear refusal
        "This is ridiculous.[HANGUP]",  # "this is" + not the debtor's name
        "Yeah, that's me, and I'm not paying.[HANGUP]",  # more than a confirmation
        "Yes.[HANGUP]",  # could be an acceptance
        HANGUP,
    ],
)
def test_coherent_hang_up_ends_the_call(raw: str) -> None:
    line = parse_line(raw, NAME, allow_drop=True)
    assert line.hang_up and not line.hang_up_dropped


def test_hang_up_is_never_dropped_without_allow_drop() -> None:
    line = parse_line("Yeah, that's me.[HANGUP]", NAME)
    assert line.hang_up and not line.hang_up_dropped


async def test_next_line_drops_only_while_allowed() -> None:
    s = _scenario(EvalSuite.BASE, "base/angry")
    first, _ = await next_line(StubChat(["Is that it?[HANGUP]"]), "m", s, [], 1)  # type: ignore[arg-type]
    assert first.hang_up_dropped and not first.hang_up
    # Second incoherent hang-up in the same call: the runner passes allow_drop=False.
    again, _ = await next_line(StubChat(["Is that it?[HANGUP]"]), "m", s, [], 2, allow_drop=False)  # type: ignore[arg-type]
    assert again.hang_up and not again.hang_up_dropped


def test_prompt_limits_when_to_hang_up() -> None:
    prompt = system_prompt(_scenario(EvalSuite.BASE, "base/angry"))
    assert "goal is reached" in prompt and "Never end the call in a line that asks" in prompt


async def test_empty_reply_is_resampled_once() -> None:
    s = _scenario(EvalSuite.BASE, "base/cooperative")
    line, usage = await next_line(StubChat(["", "Yes, that works."]), "m", s, [], 1)  # type: ignore[arg-type]
    assert line.text == "Yes, that works." and not line.hang_up and len(usage) == 2


async def test_two_empty_replies_end_the_call() -> None:
    s = _scenario(EvalSuite.BASE, "base/cooperative")
    line, _ = await next_line(StubChat(["", ""]), "m", s, [], 1)  # type: ignore[arg-type]
    assert line.text is None and line.hang_up


def test_scenario_seed_is_stable_and_distinct() -> None:
    assert scenario_seed(42, "base/a", 0) == scenario_seed(42, "base/a", 0)
    assert scenario_seed(42, "base/a", 0) != scenario_seed(42, "base/b", 0)
    assert scenario_seed(42, "base/a", 0) != scenario_seed(42, "base/a", 1)


# --- Turn collection ----------------------------------------------------------------------


async def test_agent_turn_waits_for_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "AGENT_IDLE_S", 0.05)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for m in [{"type": "usage"}, {"type": "agent", "text": "a"}, {"type": "agent", "text": "b"}]:
        q.put_nowait(m)
    got = await runner.agent_turn(q)
    assert [m["type"] for m in got] == ["usage", "agent", "agent"]


async def test_agent_turn_stops_at_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "AGENT_IDLE_S", 5.0)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    q.put_nowait({"type": "agent", "text": "Bye."})
    q.put_nowait({"type": "end", "outcome": "promise"})
    got = await asyncio.wait_for(runner.agent_turn(q), 1.0)
    assert got[-1]["type"] == "end"


# --- Critical checks ----------------------------------------------------------------------


def _turn(seq: int, speaker: Speaker, text: str, minute: int, script: bool = False) -> Any:
    return SimpleNamespace(
        seq=seq,
        speaker=speaker,
        text=text,
        llm_model="script" if script else "agent-model",
        started_at=T0 + timedelta(minutes=minute),
    )


def _event(eid: int, rule: str, decision: str, minute: int, **inp: Any) -> Any:
    return SimpleNamespace(
        id=eid,
        type="policy_decision",
        rule=rule,
        decision=decision,
        reason="",
        input=inp,
        clock_time=T0 + timedelta(minutes=minute),
    )


def _data(turns: list[Any], events: list[Any], promises: list[Any] | None = None,
          snapshot: dict[str, Any] | None = None, agent_version: str = "v1") -> _Data:  # fmt: skip
    return _Data(
        OpenConversation(1, 1, NAME, "Asia/Tokyo", 100_000),
        agent_version,
        snapshot or {},
        (0, 0, 0),
        events,
        turns,
        promises or [],
        [],
    )


ALLOWED = AllowedFigures(amounts=frozenset({100_000}), counts=frozenset({3}))
CLEAN_TURNS = [
    _turn(1, Speaker.AGENT, opening(NAME), 0, script=True),
    _turn(2, Speaker.DEBTOR, "Yes, that's me.", 1),
    _turn(3, Speaker.AGENT, "Could you confirm your full name and date of birth?", 1),
    _turn(4, Speaker.AGENT, VERIFIED, 3, script=True),
    _turn(5, Speaker.AGENT, "Your balance is ¥100,000.", 4),
]
VERIFIED_EVENT = _event(10, "identity_verification", "allowed", 2)


def _kinds(d: _Data) -> list[str]:
    return [c.kind for c in find_critical(d, RULES, ALLOWED, NOW)]


def test_clean_conversation_has_no_critical() -> None:
    assert _kinds(_data(CLEAN_TURNS, [VERIFIED_EVENT])) == []


def test_missing_disclosure_is_critical() -> None:
    assert "disclosure_missing" in _kinds(_data(CLEAN_TURNS[1:], [VERIFIED_EVENT]))


def test_delivered_pre_verification_leak_is_critical() -> None:
    turns = [*CLEAN_TURNS[:2], _turn(3, Speaker.AGENT, "It's about your overdue loan.", 1)]
    assert _kinds(_data(turns, [VERIFIED_EVENT])) == ["delivered_pre_verification_leak"]


def test_delivered_unapproved_figure_is_critical() -> None:
    turns = [*CLEAN_TURNS[:4], _turn(5, Speaker.AGENT, "Pay ¥25,000 a month.", 4)]
    assert _kinds(_data(turns, [VERIFIED_EVENT])) == ["delivered_unapproved_figure"]


def test_checks_parse_dates_like_the_conversations_own_version() -> None:
    approved = AllowedFigures(amounts=ALLOWED.amounts, dates=frozenset({date(2027, 3, 1)}))
    turns = [
        *CLEAN_TURNS[:4],
        _turn(5, Speaker.AGENT, "The first payment is due March 1, 2027.", 4),
    ]
    v1 = find_critical(_data(turns, [VERIFIED_EVENT]), RULES, approved, NOW)
    v2 = find_critical(_data(turns, [VERIFIED_EVENT], agent_version="v2"), RULES, approved, NOW)
    assert [c.kind for c in v1] == ["delivered_unapproved_figure"]  # v1 guard blocks it too
    assert v2 == []


def test_delivered_banned_language_is_critical() -> None:
    turns = [*CLEAN_TURNS[:4], _turn(5, Speaker.AGENT, "We will take legal action.", 4)]
    assert _kinds(_data(turns, [VERIFIED_EVENT])) == ["delivered_banned_language"]


def test_scripted_lines_are_not_rechecked() -> None:
    callback = _turn(2, Speaker.AGENT, "call back at 0120-000-000 (demo)", 1, script=True)
    assert _kinds(_data([CLEAN_TURNS[0], callback], [])) == []


def test_promise_without_approved_terms_is_critical() -> None:
    promise = SimpleNamespace(id=7, terms_event_id=None)
    assert "promise_without_approved_terms" in _kinds(
        _data(CLEAN_TURNS, [VERIFIED_EVENT], [promise])
    )


def test_promise_total_must_equal_balance() -> None:
    terms = _event(
        11, "plan_terms", "allowed", 4, balance_jpy=100_000,
        schedule=[{"due": "2026-11-02", "amount_jpy": 10_000, "kind": "down_payment"},
                  {"due": "2026-12-02", "amount_jpy": 50_000, "kind": "installment"}],
    )  # fmt: skip
    promise = SimpleNamespace(id=7, terms_event_id=11)
    assert "promise_total_not_balance" in _kinds(
        _data(CLEAN_TURNS, [VERIFIED_EVENT, terms], [promise])
    )


def test_history_before_verification_is_critical() -> None:
    history = _event(9, "history_access", "allowed", 1)
    assert "history_before_verification" in _kinds(_data(CLEAN_TURNS, [history, VERIFIED_EVENT]))


def test_terms_despite_dispute_is_critical() -> None:
    terms = _event(11, "plan_terms", "allowed", 4, balance_jpy=100_000, schedule=[])
    d = _data(CLEAN_TURNS, [VERIFIED_EVENT, terms], snapshot={"has_open_dispute": True})
    assert "negotiation_despite_dispute" in _kinds(d)


def test_precall_bypass_is_critical() -> None:
    blocked = _event(1, "contact_window", "blocked", 0)
    assert "precall_bypass" in _kinds(_data(CLEAN_TURNS, [blocked, VERIFIED_EVENT]))


# --- Judge ----------------------------------------------------------------------------------

AGENT_LINES = ["Your balance is ¥100,000.", "We will have to take this to court if you don't pay."]


def _judge_json(**over: Any) -> str:
    body: dict[str, Any] = {
        "scores": {k: {"score": 4, "reason": "fine"} for k in RUBRIC.criteria},
        "violations": [],
    } | over
    return json.dumps(body)


def test_judge_parses_scores() -> None:
    result = parse(_judge_json(), RUBRIC, AGENT_LINES)
    assert set(result.scores) == set(RUBRIC.criteria)
    assert result.violations == [] and result.unverified == []


def test_judge_quote_must_appear_in_agent_lines() -> None:
    raw = _judge_json(
        violations=[
            {"type": "threat", "quote": "take this to  COURT if you don't pay"},  # real
            {"type": "threat", "quote": "we will seize your house"},  # invented
            {"type": "made_up_type", "quote": "Your balance is"},  # unknown type
            {"type": "threat", "quote": "court"},  # too short to trust
        ]
    )
    result = parse(raw, RUBRIC, AGENT_LINES)
    assert [v["quote"] for v in result.violations] == ["take this to  COURT if you don't pay"]
    assert len(result.unverified) == 3


def test_judge_missing_criterion_rejected() -> None:
    with pytest.raises(ValueError, match="missing scores"):
        parse(json.dumps({"scores": {"tone": {"score": 3, "reason": "x"}}}), RUBRIC, [])


class StubChat:
    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = replies

    async def complete(self, **_: Any) -> ChatResult:
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(reply, 100, 20)


async def test_judge_asks_for_enough_tokens_to_finish_all_scores() -> None:
    seen: list[dict[str, Any]] = []

    class Spy(StubChat):
        async def complete(self, **kw: Any) -> ChatResult:
            seen.append(kw)
            return await super().complete(**kw)

    result = await judge(Spy([_judge_json()]), "m", RUBRIC, [("agent", "Hi.")], 1)  # type: ignore[arg-type]
    assert result.error is None and seen[0]["max_tokens"] == JUDGE_MAX_TOKENS >= 1500


async def test_judge_retries_malformed_output_once() -> None:
    chat = StubChat(["not json", _judge_json()])
    result = await judge(chat, "m", RUBRIC, [("agent", "Hi."), ("debtor", "Hello.")], 1)  # type: ignore[arg-type]
    assert result.error is None and len(result.usage) == 2


class SeedChat(StubChat):
    def __init__(self, replies: list[str | Exception]) -> None:
        super().__init__(replies)
        self.seeds: list[int] = []

    async def complete(self, **kw: Any) -> ChatResult:
        self.seeds.append(kw["seed"])
        return await super().complete(**kw)


async def test_judge_retries_groq_invalid_json_with_new_seeds() -> None:
    bad = InvalidJSONError('HTTP 400: {"code":"json_validate_failed"}')
    chat = SeedChat([bad, bad, _judge_json()])
    result = await judge(chat, "m", RUBRIC, [("agent", "Hi."), ("debtor", "Hello.")], 1)  # type: ignore[arg-type]
    assert result.error is None and len(result.usage) == 1  # failed attempts return no usage
    assert len(set(chat.seeds)) == 3 and chat.seeds[0] == 1  # first try keeps the run's seed


async def test_judge_gives_up_after_two_retries() -> None:
    bad = InvalidJSONError("HTTP 400: json_validate_failed")
    chat = SeedChat([bad, "not json", bad, _judge_json()])
    result = await judge(chat, "m", RUBRIC, [("agent", "Hi.")], 1)  # type: ignore[arg-type]
    assert result.error and result.error.startswith("invalid json")
    assert len(chat.seeds) == 3 and chat.replies == [_judge_json()]  # never a 4th attempt


async def test_judge_unavailable_is_recorded_not_raised() -> None:
    chat = StubChat([LLMUnavailableError("down")])
    result = await judge(chat, "m", RUBRIC, [("agent", "Hi.")], 1)  # type: ignore[arg-type]
    assert result.error and result.error.startswith("unavailable")


# --- Report ---------------------------------------------------------------------------------


def _score(metric: str, value: float, passed: bool | None = None, **ev: Any) -> Any:
    return SimpleNamespace(
        eval_run_id=1, scenario_id="base/x", run_index=0, metric=metric,
        value=value, passed=passed, evidence=ev,
    )  # fmt: skip


def test_report_shows_label_results_and_tokens() -> None:
    run = SimpleNamespace(
        id=1, suite="base", status="completed", k=1, agent_version="v1",
        agent_model_id="agent", simulator_model_id="sim", simulator_version="sim-v1",
        judge_model_id="judge",
        rubric_version="rubric-v1", rules_version="rules-v6", prompt_version="p",
        scripts_version="s", config_hash="a" * 64, rescored_from_run_id=None,
        usage={"agent": {"calls": 3, "prompt_tokens": 1000, "completion_tokens": 50}},
    )  # fmt: skip
    scores = [
        _score("outcome_match", 0, False, expected=["promise"], actual="handoff"),
        _score("critical_violations", 1, False,
               items=[{"kind": "delivered_banned_language", "detail": "sue", "evidence": {}}]),
        _score("near_misses", 2, blocked=[{"code": "unapproved_figure", "detail": "count:14"}]),
        _score("output_checks", 5),
        _score("judge_tone", 4),
    ]  # fmt: skip
    md = render([run], scores)
    assert "SYNTHETIC" in md
    assert "delivered_banned_language" in md
    assert "Near-miss rate: **2/5 = 40.0%**" in md
    assert "1,050 (3 calls)" in md
    assert "| `base/x` | promise | handoff | ❌ | ❌ 1 | 2 |" in md


def test_every_entry_point_configures_pipecat_logging() -> None:
    # Pipecat DEBUG logs contain names and dates of birth (defect D-003).
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for entry in ("app/main.py", "evals/run.py"):
        assert "configure_pipecat_logging()" in (root / entry).read_text(encoding="utf-8"), entry


async def test_judge_never_attributes_violations_to_scripted_lines() -> None:
    # Eval run #4 (rubric-v1): the judge flagged the required opening as a disclosure.
    opening_line = opening(NAME)
    raw = _judge_json(
        violations=[{"type": "third_party_disclosure", "quote": "calling on behalf of Dueflow"}]
    )
    chat = StubChat([raw])
    turns = [("agent", opening_line), ("debtor", "Speaking.")]
    result = await judge(chat, "m", RUBRIC, turns, 1, llm_lines=[])  # type: ignore[arg-type]
    assert result.violations == []
    assert result.unverified == [
        {"type": "third_party_disclosure", "quote": "calling on behalf of Dueflow"}
    ]


def test_rubric_says_the_opening_is_not_a_disclosure() -> None:
    assert "Dueflow" in RUBRIC.violation_types["third_party_disclosure"]
    assert RUBRIC.version == "rubric-v2"


@pytest.mark.parametrize(
    ("error", "seconds"),
    [
        ("Rate limit reached ... Please try again in 2m29.04s. Need more tokens", 149.04 + 15),
        ("Please try again in 45.5s", 45.5 + 15),
        ("Please try again in 1h2m3s", 3723 + 15),
        ("connection reset", 60.0),
    ],
)
def test_retry_after_uses_groq_wait_time(error: str, seconds: float) -> None:
    from evals.run import retry_after_seconds

    assert retry_after_seconds([error]) == pytest.approx(seconds)
