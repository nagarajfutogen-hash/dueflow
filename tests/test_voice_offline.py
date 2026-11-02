"""Day 5 voice pipeline, tested without audio, Deepgram or any LLM call."""

from typing import Any, cast

import httpx
import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent import scripts
from app.agent.live import LiveChat, can_open
from app.agent.pipeline import GuardedTextSink, take_sentences, trim_unspoken
from app.db.models import Channel, Event, Speaker, Turn
from app.db.turns import append_turn
from app.models_config import load_models
from app.policy.identity import find_dates
from app.policy.output_check import SENTENCE_SPLIT, AllowedFigures, check_output
from app.voice.latency import percentile, render, summarize, turn_latency, voice_turn_latencies
from app.voice.pipeline import SpokenTracker, VoiceBridge, VoiceTimer, for_speech
from tests.conftest import MakeDebtor
from tests.test_chat_offline import DOB, NAME, NOW, RULES, _session

# --- Sentences: voice splits a reply exactly where the output check does --------------------


@pytest.mark.parametrize(
    ("buffer", "done", "rest"),
    [
        ("Your balance is ¥12,000. You can", ["Your balance is ¥12,000."], "You can"),
        ("Due on 2026-11-01.", [], "Due on 2026-11-01."),  # no whitespace yet: may continue
        ("Hello. Is now a good time? ", ["Hello.", "Is now a good time?"], ""),
        ("  Okay! Thanks", ["Okay!"], "Thanks"),
        ("", [], ""),
    ],
)
def test_take_sentences(buffer: str, done: list[str], rest: str) -> None:
    assert take_sentences(buffer) == (done, rest)


REPLY = "Thank you. Your balance is ¥100,000. You can pay by 2026-10-31. Mr. Sato will call."


@pytest.mark.parametrize("chunk", [1, 3, 7, 1000])
def test_streamed_sentences_match_the_output_check_split(chunk: int) -> None:
    done: list[str] = []
    rest = ""
    for i in range(0, len(REPLY), chunk):
        new, rest = take_sentences(rest + REPLY[i : i + chunk])
        done += new
    assert [*done, rest] == SENTENCE_SPLIT.split(REPLY)


@pytest.mark.parametrize("verified", [True, False])
def test_per_sentence_check_says_what_the_whole_reply_check_says(verified: bool) -> None:
    """The README claim: voice says exactly what text would send (blocked: rest dropped)."""
    allowed = AllowedFigures(amounts=frozenset({100_000}), dates=frozenset())
    reply = "Thank you. Your balance is ¥100,000. Pay ¥5,000 today. Goodbye."

    def check(text: str) -> Any:
        return check_output(
            NOW, text, identity_verified=verified, allowed=allowed, rules=RULES, fallback="FB."
        )

    said: list[str] = []
    for sentence in SENTENCE_SPLIT.split(reply):
        result = check(sentence)
        said.append(result.text)
        if not result.decision.allowed:
            break
    assert " ".join(said) == check(reply).text


# --- Guarded sink, per sentence -------------------------------------------------------------


class SentenceGuard:
    def __init__(self, block: str | None = None) -> None:
        self.block = block
        self.seen: list[str] = []
        self.ended: list[tuple[int | None, int | None]] = []

    async def guard_sentence(self, sentence: str) -> tuple[str, bool]:
        self.seen.append(sentence)
        if sentence == self.block:
            return "FALLBACK.", False
        return sentence, True

    async def end_reply(self, prompt: int | None, completion: int | None) -> None:
        self.ended.append((prompt, completion))

    async def guard_and_send(self, raw: str, prompt: int | None, completion: int | None) -> str:
        raise AssertionError("voice never guards a whole reply")

    async def add_usage(self, prompt: int, completion: int) -> None:
        pass


def _reply(*chunks: str) -> list[Any]:
    return [
        LLMFullResponseStartFrame(),
        *(LLMTextFrame(text=c) for c in chunks),
        LLMFullResponseEndFrame(),
    ]


async def test_each_sentence_is_guarded_as_soon_as_it_is_complete() -> None:
    guard = SentenceGuard()
    down, _ = await run_test(
        GuardedTextSink(guard, per_sentence=True),
        frames_to_send=_reply("Hello there. Your bal", "ance is due. Bye."),
        expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame],
    )
    assert guard.seen == ["Hello there.", "Your balance is due.", "Bye."]
    assert guard.ended == [(None, None)]
    # The LLM context gets exactly what was said, once.
    assert [f.text for f in down if isinstance(f, LLMTextFrame)] == [
        "Hello there. Your balance is due. Bye."
    ]


async def test_a_blocked_sentence_ends_the_reply_early() -> None:
    guard = SentenceGuard(block="Your balance is due.")
    down, _ = await run_test(
        GuardedTextSink(guard, per_sentence=True),
        frames_to_send=_reply("Hello there. Your balance is due. Bye."),
        expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame],
    )
    assert guard.seen == ["Hello there.", "Your balance is due."]  # "Bye." never checked
    assert [f.text for f in down if isinstance(f, LLMTextFrame)] == ["Hello there. FALLBACK."]


async def test_tool_call_only_reply_says_nothing_per_sentence() -> None:
    guard = SentenceGuard()
    await run_test(
        GuardedTextSink(guard, per_sentence=True),
        frames_to_send=_reply(),
        expected_down_frames=[LLMFullResponseStartFrame, LLMFullResponseEndFrame],
    )
    assert guard.seen == [] and guard.ended == [(None, None)]


# --- Barge-in: the LLM context keeps what was heard ----------------------------------------


def test_trim_cuts_the_unheard_part_of_the_last_reply() -> None:
    msgs = [
        {"role": "assistant", "content": "Opening line."},
        {"role": "user", "content": "Yes?"},
        {"role": "assistant", "content": "Sentence one. Sentence two."},
    ]
    out = trim_unspoken(msgs, len("Sentence one."))
    assert out[:2] == msgs[:2]
    assert out[2]["content"] == "Sentence one. —"


def test_trim_drops_an_unheard_line_and_keeps_tool_messages() -> None:
    call = {"role": "assistant", "tool_calls": [{"id": "c1"}], "content": None}
    result = {"role": "tool", "tool_call_id": "c1", "content": "{}"}
    msgs = [
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "First."},
        call,
        result,
        {"role": "assistant", "content": "Never heard."},
    ]
    out = trim_unspoken(msgs, len("First."))
    assert out == msgs[:4]


def test_trim_keeps_everything_that_was_heard() -> None:
    msgs = [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "All of it."}]
    assert trim_unspoken(msgs, 500) == msgs


# --- Transport processors -------------------------------------------------------------------


def _final(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text, "debtor", "2026-10-24T03:00:00Z")


class Calls:
    def __init__(self) -> None:
        self.got: list[Any] = []

    async def __call__(self, value: Any) -> None:
        self.got.append(value)


async def test_one_user_turn_becomes_one_debtor_line() -> None:
    calls = Calls()
    down, _ = await run_test(
        VoiceBridge(calls),
        frames_to_send=[
            UserStartedSpeakingFrame(),
            _final("Yes, it's me."),
            _final("Born August 8, 1993."),
            UserStoppedSpeakingFrame(),
        ],
    )
    assert calls.got == ["Yes, it's me. Born August 8, 1993."]
    assert not any(isinstance(f, TranscriptionFrame) for f in down)  # never reaches TTS


async def test_a_late_final_transcript_still_ends_the_turn() -> None:
    calls = Calls()
    await run_test(
        VoiceBridge(calls),
        frames_to_send=[UserStartedSpeakingFrame(), UserStoppedSpeakingFrame(), _final("Okay.")],
    )
    assert calls.got == ["Okay."]


class Clock:
    def __init__(self, *times: float) -> None:
        self.times = list(times)

    def __call__(self) -> float:
        return self.times.pop(0)


def _audio() -> TTSAudioRawFrame:
    return TTSAudioRawFrame(b"\x00\x00" * 160, 16_000, 1)


async def test_ttfb_from_vad_end_to_first_tts_audio() -> None:
    calls = Calls()
    await run_test(
        VoiceTimer(calls, vad_stop_secs=0.2, now=Clock(10.0, 10.9)),
        frames_to_send=[VADUserStoppedSpeakingFrame(), _audio(), _audio()],
    )
    assert calls.got == [{"ttfb_ms": 900, "perceived_ms": 1100, "vad_stop_ms": 200}]


async def test_only_the_last_vad_stop_counts() -> None:
    calls = Calls()
    await run_test(
        VoiceTimer(calls, vad_stop_secs=0.2, now=Clock(1.0, 5.0, 5.5)),
        frames_to_send=[
            VADUserStoppedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(),
            _audio(),
        ],
    )
    assert calls.got == [{"ttfb_ms": 500, "perceived_ms": 700, "vad_stop_ms": 200}]


async def test_barge_in_reports_what_was_heard() -> None:
    calls = Calls()
    await run_test(
        SpokenTracker(calls, now=Clock(2.0, 3.5)),
        frames_to_send=[
            BotStartedSpeakingFrame(),
            TTSTextFrame("Hello there.", "sentence"),
            SleepFrame(0.05),  # the text was played before the debtor spoke over it
            InterruptionFrame(),
        ],
    )
    assert calls.got == [{"interrupted": True, "interrupted_at_ms": 1500, "spoken_chars": 12}]


async def test_interruption_while_silent_is_not_a_barge_in() -> None:
    calls = Calls()
    await run_test(
        SpokenTracker(calls, now=Clock(2.0)),
        frames_to_send=[BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(), InterruptionFrame()],
    )
    assert calls.got == []


# --- Latency report -------------------------------------------------------------------------


def test_turn_latency_rounds_to_whole_ms() -> None:
    assert turn_latency(1.0, 1.2345, 0.2) == {
        "ttfb_ms": 234,
        "perceived_ms": 434,
        "vad_stop_ms": 200,
    }


def test_nearest_rank_percentiles() -> None:
    values = [float(v) for v in range(1, 11)]
    assert (percentile(values, 50), percentile(values, 95)) == (5.0, 10.0)
    assert percentile([700.0], 95) == 700.0
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summary_reports_n_and_skips_turns_without_the_metric() -> None:
    rows: list[dict[str, Any]] = [
        {"ttfb_ms": 800, "perceived_ms": 1000},
        {"ttfb_ms": 1200, "perceived_ms": 1400},
        {"interrupted": True},
    ]
    by_metric = {s.metric: s for s in summarize(rows)}
    assert by_metric["ttfb_ms"].n == 2 and by_metric["ttfb_ms"].p50 == 800
    text = render(summarize(rows), site="local", labels={"HUMAN_TEST"}, interrupted=1)
    assert "site: local" in text and "HUMAN_TEST" in text and "Barge-ins recorded: 1" in text
    assert "No measured voice turns." in render([], site="vps", labels=set(), interrupted=0)


# --- Session and API (real DB, no LLM) ------------------------------------------------------


async def test_guard_sentence_grows_one_turn_and_audits_each_sentence(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, _ = await _session(api_client, make_debtor, app_engine)
    assert await session.guard_sentence("Thanks for picking up.") == (
        "Thanks for picking up.",
        True,
    )
    text, allowed = await session.guard_sentence("This is about your debt.")  # not verified
    assert (text, allowed) == (scripts.SAFE_FALLBACK_PRE, False)
    await session.end_reply(120, 30)
    cid = session.conversation_id
    async with app_engine.connect() as conn:
        turns = (await conn.execute(select(Turn).where(Turn.conversation_id == cid))).all()
        checks = await conn.scalar(
            select(func.count()).where(Event.conversation_id == cid, Event.rule == "output_check")
        )
    assert len(turns) == 1
    turn = turns[0]
    assert turn.text == f"Thanks for picking up. {scripts.SAFE_FALLBACK_PRE}"
    assert (turn.prompt_tokens, turn.completion_tokens) == (120, 30)
    assert checks == 2
    assert rec.said() == ["Thanks for picking up.", scripts.SAFE_FALLBACK_PRE]


async def test_voice_timings_merge_into_the_agent_turn_being_said(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, _ = await _session(api_client, make_debtor, app_engine)
    await session.say_scripted("Hello.")
    await session.note_voice({"ttfb_ms": 900})
    await session.note_voice({"interrupted": True, "spoken_chars": 3})
    async with app_engine.connect() as conn:
        latency = await conn.scalar(
            select(Turn.latency_ms).where(
                Turn.conversation_id == session.conversation_id, Turn.seq == session.last_agent_seq
            )
        )
    assert latency == {"ttfb_ms": 900, "interrupted": True, "spoken_chars": 3}


async def _dial(client: httpx.AsyncClient, make_debtor: MakeDebtor, channel: str) -> int:
    ref = await make_debtor(full_name=NAME, dob=DOB)
    body = {
        "debtor_ref": ref,
        "sim_time": "2026-10-24T12:00:00+09:00",
        "target": "mobile",
        "agent_version": "v3",
        "channel": channel,
        "label": "HUMAN_TEST",
    }
    return int((await client.post("/calls", json=body)).json()["conversation_id"])


async def test_a_conversation_opens_only_on_its_own_channel(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    voice = await _dial(api_client, make_debtor, "voice")
    text = await _dial(api_client, make_debtor, "text")
    assert await can_open(app_engine, voice, Channel.VOICE)
    assert not await can_open(app_engine, voice, Channel.TEXT)
    assert not await can_open(app_engine, text, Channel.VOICE)

    async def send(_: dict[str, Any]) -> None:
        pass

    live = await LiveChat.start(
        conversation_id=text, engine=app_engine, rules=RULES, models=load_models(),
        api_key="offline", send=send, channel=Channel.VOICE,
    )  # fmt: skip
    assert live is None


async def test_voice_offer_refuses_a_text_conversation(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    text = await _dial(api_client, make_debtor, "text")
    offer = {"sdp": "v=0", "type": "offer"}
    res = await api_client.post(f"/voice/offer?conversation_id={text}", json=offer)
    assert res.status_code == 404
    res = await api_client.post(
        f"/voice/offer?conversation_id={text}", json=offer, headers={"Authorization": "Bearer x"}
    )
    assert res.status_code == 401


async def test_latency_report_reads_voice_turns_only(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    voice = await _dial(api_client, make_debtor, "voice")
    text = await _dial(api_client, make_debtor, "text")
    async with app_engine.begin() as conn:
        for cid, ms in ((voice, 777), (text, 999)):
            await append_turn(
                conn, conversation_id=cid, speaker=Speaker.AGENT, text="Hi.",
                started_at=NOW, latency_ms={"ttfb_ms": ms},
            )  # fmt: skip
        rows = await voice_turn_latencies(conn)
    values = [r["ttfb_ms"] for r, _ in rows]
    assert 777 in values and 999 not in values
    assert {label for _, label in rows} == {"HUMAN_TEST"}


# --- Gap 1: dates as Deepgram writes them (smart_format) -------------------------------------


@pytest.mark.parametrize(
    "heard", ["It's Sora Chattest, born August 8, 1993.", "August 8th, 1993", "8 August 1993"]
)
def test_smart_formatted_dates_are_read_by_identity(heard: str) -> None:
    assert DOB in find_dates(heard)


def test_dates_left_in_words_are_read() -> None:
    """V5-7 (live VPS call 3): smart_format left the date in words; now read."""
    assert DOB in find_dates("August eighth nineteen ninety three")


def test_ambiguous_dates_are_not_read() -> None:
    """Day/month order unknown: never guessed, the call asks again (not an attempt)."""
    assert DOB not in find_dates("08/08/1993")


# --- Wiring: the real pipeline builds (no network: nothing connects until a call runs) -------


def test_voice_call_builds_with_vad_turns_stt_and_tts(test_settings: Any) -> None:
    from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

    from app.voice.pipeline import VoiceCall

    call = VoiceCall(
        SmallWebRTCConnection(),
        conversation_id=1,
        engine=cast(AsyncEngine, None),  # not touched until the client connects
        rules=RULES,
        models=load_models(),
        settings=test_settings,
    )

    def flat(p: Any) -> list[Any]:
        return [p, *(q for child in getattr(p, "processors", []) for q in flat(child))]

    processors = flat(call.worker.pipeline)
    names = [type(p).__name__ for p in processors]
    order = ["VADProcessor", "DeepgramSTTService", "UserTurnProcessor", "VoiceBridge",
             "DeepgramTTSService", "VoiceTimer", "SpokenTracker"]  # fmt: skip
    assert [n for n in names if n in order] == order
    turns = next(p for p in processors if type(p).__name__ == "UserTurnProcessor")
    stops = turns._user_turn_controller._user_turn_strategies.stop  # Smart Turn is dropped
    assert [type(s).__name__ for s in stops] == ["SpeechTimeoutUserTurnStopStrategy"]


# --- Amounts are read as yen (voice only) ---------------------------------------------------


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("Your balance is ¥12,000.", "Your balance is 12,000 yen."),
        ("Pay ¥ 5,000 now, then JPY 1,000,000.", "Pay 5,000 yen now, then 1,000,000 yen."),
        ("That is 30,000 yen per month.", "That is 30,000 yen per month."),  # already spoken
        ("No amounts here.", "No amounts here."),
    ],
)
def test_for_speech_reads_amounts_as_yen(text: str, spoken: str) -> None:
    assert for_speech(text) == spoken


async def test_tts_gets_spoken_amounts_while_the_record_keeps_the_guarded_text(
    test_settings: Any,
) -> None:
    from pipecat.frames.frames import TTSSpeakFrame
    from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

    from app.voice.pipeline import VoiceCall

    call = VoiceCall(
        SmallWebRTCConnection(),
        conversation_id=1,
        engine=cast(AsyncEngine, None),
        rules=RULES,
        models=load_models(),
        settings=test_settings,
    )
    queued: list[Any] = []

    async def queue_frames(frames: list[Any]) -> None:
        queued.extend(frames)

    call.worker.queue_frames = queue_frames  # type: ignore[method-assign,assignment]
    message = {"type": "agent", "source": "llm", "text": "Your balance is ¥12,000."}
    await call.send(message)
    assert [f.text for f in queued if isinstance(f, TTSSpeakFrame)] == [
        "Your balance is 12,000 yen."
    ]
    assert message["text"] == "Your balance is ¥12,000."  # what the turn and events hold


def test_voice_stt_gets_the_debtors_name_as_keyterms(test_settings: Any) -> None:
    """V5-7 (live call 3): nova-3 heard a first name as "data"; the name words are keyterms."""
    from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

    from app.voice.pipeline import VoiceCall

    call = VoiceCall(
        SmallWebRTCConnection(),
        conversation_id=1,
        engine=cast(AsyncEngine, None),
        rules=RULES,
        models=load_models(),
        settings=test_settings,
        keyterms=["Arun", "Kumar"],
    )
    stack: list[Any] = [call.worker.pipeline]
    stt: Any = None
    while stack:
        p = stack.pop()
        if type(p).__name__ == "DeepgramSTTService":
            stt = p
        stack.extend(getattr(p, "processors", []))
    assert stt is not None
    assert stt._settings.keyterm == ["Arun", "Kumar"]


async def test_name_keyterms_come_from_the_debtor_on_file(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    from app.voice.pipeline import name_keyterms

    session, _, _ = await _session(api_client, make_debtor, app_engine, "v5")
    assert await name_keyterms(app_engine, session.conversation_id) == NAME.split()
