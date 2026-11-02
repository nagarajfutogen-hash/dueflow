"""Phase 5 tests that make no LLM calls (the Groq key is a fake value here)."""

import asyncio
import uuid
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, date, datetime
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient
from pipecat.flows import NO_RESPONSE, FlowManager
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.tests.utils import run_test
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.websockets import WebSocketDisconnect

from app.agent import scripts
from app.agent.nodes import identify_node, negotiate_node
from app.agent.pipeline import GuardedTextSink
from app.agent.session import ChatSession
from app.agent.tools import ASK_ACCEPT, Handler, make_handlers
from app.agent.versions import load_version
from app.clock import FixedClock, OffsetClock
from app.config import Settings
from app.db.models import Conversation, Event, Outcome, PromiseStatus, Turn
from app.db.seed import SeedDebtor, SeedPromise, insert_debtor
from app.db.session import make_engine
from app.main import app
from app.policy.config import load_rules
from tests.conftest import TEST_API_TOKEN, MakeDebtor

RULES = load_rules()
NAME = "Sora Chattest"
DOB = date(1993, 8, 8)
NOW = datetime(2026, 10, 24, 3, tzinfo=UTC)
FM = cast(FlowManager, None)  # handlers never touch the flow manager


# --- Guarded sink ---------------------------------------------------------------------------


class StubGuard:
    def __init__(self) -> None:
        self.seen: list[str] = []

    async def guard_and_send(self, raw: str, prompt: int | None, completion: int | None) -> str:
        self.seen.append(raw)
        return "GUARDED."

    async def add_usage(self, prompt: int, completion: int) -> None:
        pass

    async def guard_sentence(self, sentence: str) -> tuple[str, bool]:
        raise AssertionError("text mode never guards per sentence")

    async def end_reply(self, prompt: int | None, completion: int | None) -> None:
        raise AssertionError("text mode never ends a voice reply")


async def test_sink_replaces_streamed_text_with_guarded_reply() -> None:
    guard = StubGuard()
    down, _ = await run_test(
        GuardedTextSink(guard),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame(text="About your "),
            LLMTextFrame(text="debt."),
            LLMFullResponseEndFrame(),
        ],
        expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame],
    )
    assert guard.seen == ["About your debt."]
    texts = [f.text for f in down if isinstance(f, LLMTextFrame)]
    assert texts == ["GUARDED."]  # the raw text never goes downstream


async def test_sink_tool_call_only_response_sends_nothing() -> None:
    guard = StubGuard()
    await run_test(
        GuardedTextSink(guard),
        frames_to_send=[LLMFullResponseStartFrame(), LLMFullResponseEndFrame()],
        expected_down_frames=[LLMFullResponseStartFrame, LLMFullResponseEndFrame],
    )
    assert guard.seen == []


# --- Prompts ------------------------------------------------------------------------------


def test_identify_prompt_has_no_dob_or_full_company() -> None:
    node = identify_node(NAME, load_version("v1"))
    assert "1993" not in node.task and "Servicing" not in node.task
    assert node.respond_immediately is False


def test_negotiate_prompt_uses_broken_promise_minimum() -> None:
    node = negotiate_node(
        {"balance_jpy": 100_000, "has_broken_promise": True}, RULES, load_version("v1")
    )
    assert "100000" in node.task
    assert f"{RULES.plan.min_down_payment_pct_after_broken_promise}% down" in node.task


def test_offset_clock_advances_from_sim_start() -> None:
    real = FixedClock(datetime(2030, 1, 1, tzinfo=UTC))
    clock = OffsetClock(NOW, real)
    assert clock.now() == NOW


# --- Tool handlers (real DB, no LLM) ------------------------------------------------------


class Recorder:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    def said(self) -> list[str]:
        return [m["text"] for m in self.sent if m["type"] == "agent"]


async def _session(
    client: httpx.AsyncClient,
    make_debtor: MakeDebtor,
    engine: AsyncEngine,
    version: str = "v1",
    **debtor: Any,
) -> tuple[ChatSession, Recorder, dict[str, Handler]]:
    ref = await make_debtor(full_name=NAME, dob=DOB, **debtor)
    body = {
        "debtor_ref": ref,
        "sim_time": "2026-10-24T12:00:00+09:00",
        "target": "mobile",
        "agent_version": version,
    }
    cid = int((await client.post("/calls", json=body)).json()["conversation_id"])
    rec = Recorder()
    session = ChatSession(
        conversation_id=cid,
        full_name=NAME,
        agent_version=version,
        model_id="test-model",
        engine=engine,
        rules=RULES,
        clock=FixedClock(NOW),
        send=rec,
    )
    return session, rec, make_handlers(session)


async def _verify(
    session: ChatSession, h: dict[str, Handler], dob: str = "1993-08-08"
) -> tuple[dict[str, Any], Any]:
    await session.record_debtor(f"It's {NAME}, born {dob}.")
    return await h["verify_identity"]({}, FM)


async def test_wrong_dob_retries_with_scripted_line(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine)
    result, nxt = await _verify(session, h, "1993-08-09")
    assert result["status"] == "retry" and nxt is NO_RESPONSE
    assert rec.said() == [scripts.VERIFY_RETRY]


async def test_verified_moves_to_negotiate_with_facts(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine)
    result, nxt = await _verify(session, h)
    assert result["status"] == "verified"
    assert nxt["name"] == "negotiate"
    assert "100000" in nxt["task_messages"][0]["content"]
    assert rec.said() == [scripts.VERIFIED]
    assert session.node == "negotiate"


async def test_verified_with_open_dispute_hands_off(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine, open_dispute="paid")
    _, nxt = await _verify(session, h)
    assert nxt is NO_RESPONSE
    assert rec.said() == [scripts.VERIFIED, scripts.HANDOFF_END]
    assert session.outcome == Outcome.HANDOFF.value and session.ended.is_set()


async def test_third_party_ends_with_callback(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine)
    await h["report_third_party"]({"kind": "not_available"}, FM)
    assert rec.said() == [scripts.third_party_callback(NAME)]
    assert session.outcome == Outcome.THIRD_PARTY_CALLBACK.value


async def test_approved_terms_are_said_by_code_then_promise_ends_call(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine)
    await _verify(session, h)
    result, nxt = await h["propose_terms"]({"months": 3, "down_payment_jpy": 10_000}, FM)
    assert result["approved"] is True and nxt is NO_RESPONSE
    assert rec.said()[-1] == f"{result['already_said_to_debtor']} {ASK_ACCEPT}"

    bad, bad_next = await h["record_promise"]({"terms_id": 1}, FM)
    assert bad["recorded"] is False and bad_next is None
    done, _ = await h["record_promise"]({"terms_id": result["terms_id"]}, FM)
    assert done["recorded"] is True
    assert session.outcome == Outcome.PROMISE.value


async def test_refused_terms_go_back_to_llm(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(
        api_client,
        make_debtor,
        app_engine,
        promises=(SeedPromise(3, 20_000, 30_000, date(2026, 8, 1), PromiseStatus.BROKEN),),
    )
    await _verify(session, h)
    result, nxt = await h["propose_terms"]({"months": 3, "down_payment_jpy": 10_000}, FM)
    assert result["approved"] is False
    assert result["refusal_codes"] == ["down_payment_too_low"]
    assert nxt is None  # LLM explains, through the interceptor


async def test_tools_do_nothing_after_call_ended(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine)
    await h["record_stop_contact"]({}, FM)
    result, nxt = await h["propose_terms"]({"months": 3, "down_payment_jpy": 10_000}, FM)
    assert result == {"status": "call already ended"} and nxt is NO_RESPONSE


# --- WebSocket (TestClient runs its own event loop: use a fresh engine) ----------------------


@pytest.fixture
def ws_client(test_settings: Settings) -> Iterator[TestClient]:
    # One TestClient context = one event loop for every request, so pooled DB connections
    # stay valid. Lifespan points state at the dev DB; switch it to the test DB inside.
    with TestClient(app) as client:
        app.state.settings = replace(
            test_settings, api_token=TEST_API_TOKEN, groq_api_key="offline"
        )
        app.state.engine = make_engine(test_settings.app_url)
        app.state.rules = RULES
        client.headers["Authorization"] = f"Bearer {TEST_API_TOKEN}"
        yield client


def _hang_up(ws: Any) -> dict[str, Any]:
    """Hang up and wait for the server to finish: no race with TestClient's teardown."""
    ws.send_json({"type": "hangup"})
    end = ws.receive_json()
    with pytest.raises(WebSocketDisconnect):
        ws.receive_json()
    return dict(end)


def _dial(client: TestClient, ref: str) -> int:
    body = {
        "debtor_ref": ref,
        "sim_time": "2026-10-24T12:00:00+09:00",
        "target": "mobile",
        "agent_version": "v1",
    }
    return int(client.post("/calls", json=body).json()["conversation_id"])


def _new_debtor() -> str:
    """Sync tests can't use async fixtures: insert with a short-lived engine of their own."""
    ref = f"T-{uuid.uuid4().hex[:10]}"

    async def _insert() -> None:
        engine = make_engine(app.state.settings.owner_url)
        async with engine.begin() as conn:
            await insert_debtor(
                conn,
                SeedDebtor(ref=ref, persona="test", full_name=NAME, dob=DOB, balance_jpy=100_000),
            )
        await engine.dispose()

    asyncio.run(_insert())
    return ref


def test_ws_rejects_bad_token(ws_client: TestClient) -> None:
    cid = _dial(ws_client, _new_debtor())
    with ws_client.websocket_connect(f"/chat/ws?conversation_id={cid}") as ws:
        ws.send_json({"type": "auth", "token": "wrong"})
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.code == 1008


def test_ws_rejects_unknown_conversation(ws_client: TestClient) -> None:
    with ws_client.websocket_connect("/chat/ws?conversation_id=999999999") as ws:
        ws.send_json({"type": "auth", "token": TEST_API_TOKEN})
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.reason == "conversation not available"


def test_ws_sends_opening_by_code_and_logs_disclosure(ws_client: TestClient) -> None:
    cid = _dial(ws_client, _new_debtor())
    with ws_client.websocket_connect(f"/chat/ws?conversation_id={cid}") as ws:
        ws.send_json({"type": "auth", "token": TEST_API_TOKEN})
        first = ws.receive_json()
        end = _hang_up(ws)
    assert first == {"type": "agent", "source": "script", "text": scripts.opening(NAME)}
    assert end == {"type": "end", "outcome": "disconnected"}

    async def _check() -> tuple[list[str], list[Any]]:
        engine = make_engine(app.state.settings.app_url)
        async with engine.connect() as conn:
            turns = list(
                (await conn.execute(select(Turn.text).where(Turn.conversation_id == cid))).scalars()
            )
            events = list(
                (
                    await conn.execute(
                        select(Event.input).where(
                            Event.conversation_id == cid, Event.type == "state_transition"
                        )
                    )
                ).scalars()
            )
        await engine.dispose()
        return turns, events

    turns, events = asyncio.run(_check())
    assert turns == [scripts.opening(NAME)]
    assert {"to": "VerifyIdentity"} in events
    assert {"to": "End", "outcome": "disconnected"} in events


def test_ws_second_session_for_same_conversation_rejected(ws_client: TestClient) -> None:
    cid = _dial(ws_client, _new_debtor())
    with ws_client.websocket_connect(f"/chat/ws?conversation_id={cid}") as ws:
        ws.send_json({"type": "auth", "token": TEST_API_TOKEN})
        ws.receive_json()
        _hang_up(ws)
    with ws_client.websocket_connect(f"/chat/ws?conversation_id={cid}") as ws:
        ws.send_json({"type": "auth", "token": TEST_API_TOKEN})
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.reason == "conversation not available"


async def test_llm_supplied_identity_values_are_ignored(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine)
    await session.record_debtor(f"Yes, this is {NAME}.")
    # An LLM inventing the right-looking values must not verify anyone or burn an attempt.
    result, nxt = await h["verify_identity"]({"full_name": NAME, "date_of_birth": "1993-08-08"}, FM)
    assert result["status"] == "need_details" and nxt is None
    assert rec.said() == []
    # The person then states their details: verified on the first counted attempt.
    result, _ = await _verify(session, h)
    assert result["status"] == "verified"


async def _conversation_row(engine: AsyncEngine, cid: int) -> Any:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                select(
                    Conversation.outcome,
                    Conversation.llm_calls,
                    Conversation.prompt_tokens,
                    Conversation.completion_tokens,
                ).where(Conversation.id == cid)
            )
        ).one()


async def test_identify_turn_cap_ends_with_no_verification(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, _ = await _session(api_client, make_debtor, app_engine)
    cap = RULES.identity.max_turns_without_attempt
    for i in range(cap - 1):
        assert await session.record_debtor(f"Who is this? ({i})") is False
    assert await session.record_debtor("I'm not telling you anything.") is True
    assert rec.said() == [scripts.NO_VERIFICATION_END]
    assert session.outcome == Outcome.NO_VERIFICATION.value and session.ended.is_set()
    row = await _conversation_row(app_engine, session.conversation_id)
    assert row.outcome is Outcome.NO_VERIFICATION


async def test_counted_attempt_resets_the_turn_cap(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, h = await _session(api_client, make_debtor, app_engine)
    cap = RULES.identity.max_turns_without_attempt
    for _ in range(cap - 2):
        await session.record_debtor("Hello?")
    result, _ = await _verify(session, h, "1993-08-09")  # wrong date: a counted attempt
    assert result["status"] == "retry"
    for _ in range(cap - 1):
        assert await session.record_debtor("Hmm.") is False
    assert not session.ended.is_set()


async def test_turn_cap_only_applies_before_verification(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, _, h = await _session(api_client, make_debtor, app_engine)
    await _verify(session, h)
    for _ in range(RULES.identity.max_turns_without_attempt + 2):
        assert await session.record_debtor("Let me think about it.") is False


async def test_usage_saved_per_conversation_including_tool_only_calls(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    session, rec, _ = await _session(api_client, make_debtor, app_engine)
    await session.add_usage(900, 20)  # e.g. a tool-call-only reply: no turn is written
    await session.add_usage(1100, 60)
    row = await _conversation_row(app_engine, session.conversation_id)
    assert (row.llm_calls, row.prompt_tokens, row.completion_tokens) == (2, 2000, 80)
    assert rec.sent[-1] == {
        "type": "usage",
        "llm_calls": 2,
        "prompt_tokens": 2000,
        "completion_tokens": 80,
    }


def test_agent_llm_follows_the_provider() -> None:
    from pipecat.services.groq.llm import GroqLLMService
    from pipecat.services.openai.llm import OpenAILLMService

    from app.agent.pipeline import make_llm

    v4 = load_version("v4")
    assert type(make_llm("openai", "k", "gpt-4.1-mini", v4)) is OpenAILLMService
    assert type(make_llm("groq", "k", "openai/gpt-oss-20b", v4)) is GroqLLMService
