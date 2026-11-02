"""Agent versions: v1 stays exactly the measured baseline, v2 changes only behind its switches."""

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pipecat.flows import FlowManager
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame
from pipecat.pipeline.worker import PipelineWorker
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent import scripts
from app.agent.live import LiveChat
from app.agent.nodes import _ROLE, identify_node, negotiate_node
from app.agent.scripts import NEUTRAL_COMPANY_NAME
from app.agent.versions import (
    UnknownAgentVersionError,
    load_version,
    load_version_file,
)
from app.db.models import Dispute, Event, Outcome
from app.policy.config import load_rules
from tests.conftest import MakeDebtor
from tests.test_chat_offline import NAME, Recorder, _session

RULES = load_rules()
V1, V2 = load_version("v1"), load_version("v2")

# The identify prompt exactly as measured in the v1 baseline (prompts-v2).
V1_IDENTIFY_TASK = (
    f"The call opened by asking for {NAME}. Your caller name is '{NEUTRAL_COMPANY_NAME}' only. "
    "Until identity is verified, never mention debt, money, payments, the account, or why you "
    "are calling.\n"
    f"- If the person says they are {NAME}: ask for their full name and date of birth, then "
    "call verify_identity (date as YYYY-MM-DD).\n"
    f"- If {NAME} is not available: call report_third_party(not_available).\n"
    "- If they don't know that person: call report_third_party(wrong_number).\n"
    "- If they ask not to be called again: call record_stop_contact.\n"
    f"- If asked why you are calling: say it is a personal matter for {NAME} only."
)


# --- Version files --------------------------------------------------------------------------


def test_v1_is_the_baseline_with_every_fix_off() -> None:
    assert V1.prompt_version == "prompts-v2"
    assert not any(V1.fixes.model_dump().values())


V2_FIXES = ("code_identity_check", "wrong_number_guidance", "empathy_prompt", "full_date_parsing")
V3_FIXES = ("verified_wrong_number", "payment_claim_handoff")
V3 = load_version("v3")


def test_v2_turns_on_its_four_fixes_only() -> None:
    assert V2.prompt_version != V1.prompt_version
    assert all(getattr(V2.fixes, f) for f in V2_FIXES)
    assert not any(getattr(V2.fixes, f) for f in V3_FIXES)
    # Same sampling and rules: v2 differs from v1 only by its fixes.
    assert (V2.temperature, V2.reasoning_effort, V2.rules_version) == (
        V1.temperature,
        V1.reasoning_effort,
        V1.rules_version,
    )


V4_FIXES = (
    "llm_presents_terms",
    "affordability_first",
    "refusal_wording",
    "warm_scripts",
    "retry_blocked_reply",
    "no_repeat_offers",
    "customer_context",
    "llm_goodbyes",
    "approaches",
    "payment_check",
    "echo_relative_dates",
)


def test_v3_is_v2_plus_two_code_fixes() -> None:
    assert all(getattr(V3.fixes, f) for f in (*V2_FIXES, *V3_FIXES))
    assert not any(getattr(V3.fixes, f) for f in V4_FIXES)
    assert (V3.prompt_version, V3.temperature, V3.reasoning_effort) == (
        V2.prompt_version,
        V2.temperature,
        V2.reasoning_effort,
    )


def test_versions_name_their_rules_file() -> None:
    # v1/v2 were measured on rules-v6 and moved to rules-v7 on 2026-10-27; v1-v4 moved to
    # rules-v8 on 2026-10-31 (identical decisions both times: tests/test_rules_versions.py), so
    # every version compares strictly on the same rules.
    v4 = load_version("v4")
    assert V1.rules_version == V2.rules_version == V3.rules_version == v4.rules_version
    assert v4.rules_version == RULES.version == "rules-v8"


@pytest.mark.parametrize("name", ["v9", "../config/models", ""])
def test_unknown_version_is_rejected(name: str) -> None:
    with pytest.raises(UnknownAgentVersionError):
        load_version_file(name)


def test_file_name_must_match_declared_name(tmp_path: Path) -> None:
    (tmp_path / "v3.yaml").write_text(
        Path("agent_versions/v2.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="declares name"):
        load_version_file("v3", tmp_path)


async def test_dial_with_unknown_version_is_422(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    body = {
        "debtor_ref": await make_debtor(),
        "sim_time": "2026-10-24T12:00:00+09:00",
        "target": "mobile",
    }
    assert (await api_client.post("/calls", json=body | {"agent_version": "v9"})).status_code == 422
    assert (await api_client.post("/calls", json=body | {"agent_version": "v2"})).status_code == 200


# --- Prompts ----------------------------------------------------------------------------------


def test_v1_prompts_are_unchanged() -> None:
    node = identify_node(NAME, V1)
    assert node.role == _ROLE
    assert node.task == V1_IDENTIFY_TASK
    assert negotiate_node({"balance_jpy": 100_000}, RULES, V1).role == _ROLE


def test_v2_prompts_add_empathy_and_wrong_number_guidance() -> None:
    node = identify_node(NAME, V2)
    assert node.role.startswith(_ROLE) and "acknowledge it" in node.role
    assert "Being annoyed, busy or asking why you are calling is not a wrong number" in node.task
    assert "- If they don't know that person:" not in node.task
    assert "acknowledge it" in negotiate_node({"balance_jpy": 100_000}, RULES, V2).role


def test_v2_identify_prompt_still_reveals_nothing() -> None:
    task = identify_node(NAME, V2).task
    assert "1993" not in task and "Servicing" not in task


# --- Code-run identity check (fixes.code_identity_check) ------------------------------------


class StubWorker:
    def __init__(self, log: list[str]) -> None:
        self.frames: list[Frame] = []
        self.log = log

    async def queue_frames(self, frames: list[Frame]) -> None:
        self.frames += frames
        self.log += [f"frame:{role}" for role, _ in self._roles(frames)]

    @staticmethod
    def _roles(frames: list[Frame]) -> list[tuple[str, bool]]:
        return [
            (cast(dict[str, Any], m)["role"], bool(f.run_llm))
            for f in frames
            if isinstance(f, LLMMessagesAppendFrame)
            for m in f.messages
        ]

    def appended(self) -> list[tuple[str, bool]]:
        """(role, run_llm) of every message appended to the LLM context, in order."""
        return self._roles(self.frames)

    def tool_results(self) -> list[dict[str, Any]]:
        return [
            json.loads(cast(dict[str, Any], m)["content"])
            for f in self.frames
            if isinstance(f, LLMMessagesAppendFrame)
            for m in f.messages
            if cast(dict[str, Any], m)["role"] == "tool"
        ]


class StubFlow:
    def __init__(self, log: list[str]) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.log = log

    async def set_node_from_config(self, node: dict[str, Any]) -> None:
        self.nodes.append(node)
        self.log.append(f"node:{node['name']}")


async def _live(
    client: httpx.AsyncClient, make_debtor: MakeDebtor, engine: AsyncEngine, version: str
) -> tuple[LiveChat, StubWorker, StubFlow, Recorder]:
    session, rec, handlers = await _session(client, make_debtor, engine, version)
    log: list[str] = []
    worker, flow = StubWorker(log), StubFlow(log)
    live = LiveChat(
        session,
        cast(PipelineWorker, worker),
        cast(FlowManager, flow),
        handlers,
        cast("asyncio.Task[Any]", None),
    )
    return live, worker, flow, rec


async def _identity_decisions(engine: AsyncEngine, cid: int) -> list[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            select(Event.decision)
            .where(Event.conversation_id == cid, Event.rule == "identity_verification")
            .order_by(Event.id)
        )
        return [r.decision.value for r in rows]


async def test_v2_correct_details_are_verified_by_code(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    live, worker, flow, rec = await _live(api_client, make_debtor, app_engine, "v2")
    await live.debtor_says(f"{NAME}, 1993-08-08")
    assert await _identity_decisions(app_engine, live.session.conversation_id) == ["allowed"]
    assert rec.said() == [scripts.VERIFIED]
    assert [n["name"] for n in flow.nodes] == ["negotiate"]
    # The LLM runs only from the negotiate node, never on the raw name + date of birth.
    assert worker.appended() == [("user", False), ("assistant", False), ("tool", False)]


async def test_v2_code_check_leaves_the_same_context_as_the_tool_path(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    # D-006 regression: without a verify_identity call + result in the context, the LLM read
    # the debtor's name + date of birth as a plan and called propose_terms(0, 0).
    live, worker, flow, _ = await _live(api_client, make_debtor, app_engine, "v2")
    await live.debtor_says(f"{NAME}, 1993-08-08")
    assert flow.log == ["frame:user", "frame:assistant", "frame:tool", "node:negotiate"]
    assert worker.tool_results() == [{"status": "verified"}]
    call = cast(dict[str, Any], worker.frames[1].messages[0])["tool_calls"][0]  # type: ignore[attr-defined]
    assert call["function"] == {"name": "verify_identity", "arguments": "{}"}


async def test_v2_wrong_details_are_a_counted_attempt(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    live, worker, flow, rec = await _live(api_client, make_debtor, app_engine, "v2")
    await live.debtor_says(f"{NAME}, born 1993-08-09")
    assert await _identity_decisions(app_engine, live.session.conversation_id) == ["blocked"]
    assert rec.said() == [scripts.VERIFY_RETRY]
    assert flow.nodes == [] and live.session.node == "identify"
    # Code already said the retry line: the LLM waits for the person's next words.
    assert worker.appended() == [("user", False), ("assistant", False), ("tool", False)]
    assert worker.tool_results() == [{"status": "retry", "attempts_left": 1}]


async def test_v2_without_a_date_leaves_it_to_the_llm(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    live, worker, flow, rec = await _live(api_client, make_debtor, app_engine, "v2")
    await live.debtor_says("Seriously? I'm busy.")
    assert await _identity_decisions(app_engine, live.session.conversation_id) == []
    assert worker.appended() == [("user", True)] and rec.said() == []


async def test_v1_never_runs_the_check_by_code(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    live, worker, flow, rec = await _live(api_client, make_debtor, app_engine, "v1")
    await live.debtor_says(f"{NAME}, 1993-08-08")
    assert await _identity_decisions(app_engine, live.session.conversation_id) == []
    assert worker.appended() == [("user", True)] and flow.nodes == []


async def test_v2_date_on_the_cap_turn_is_checked_not_cut_off(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    live, _, flow, _ = await _live(api_client, make_debtor, app_engine, "v2")
    for i in range(RULES.identity.max_turns_without_attempt - 1):
        await live.debtor_says(f"Who is calling? ({i})")
    await live.debtor_says(f"Fine. {NAME}, 1993-08-08.")
    assert not live.session.ended.is_set()
    assert [n["name"] for n in flow.nodes] == ["negotiate"]


async def test_v1_cap_turn_still_ends_the_call(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    live, _, _, rec = await _live(api_client, make_debtor, app_engine, "v1")
    for i in range(RULES.identity.max_turns_without_attempt - 1):
        await live.debtor_says(f"Who is calling? ({i})")
    await live.debtor_says(f"Fine. {NAME}, 1993-08-08.")
    assert live.session.ended.is_set()
    assert rec.said() == [scripts.NO_VERIFICATION_END]


# --- v3: wrong number needs the person's words (fixes.verified_wrong_number) -----------------


async def _rule_decisions(engine: AsyncEngine, cid: int, rule: str) -> list[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            select(Event.decision)
            .where(Event.conversation_id == cid, Event.rule == rule)
            .order_by(Event.id)
        )
        return [r.decision.value for r in rows]


async def test_v3_unstated_wrong_number_does_not_end_the_call(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    live, _, _, rec = await _live(api_client, make_debtor, app_engine, "v3")
    await live.debtor_says("What do you want?")
    result, nxt = await live.handlers["report_third_party"]({"kind": "wrong_number"}, live.flow)
    assert result["status"] == "not_confirmed" and nxt is None  # the LLM asks instead
    assert not live.session.ended.is_set() and rec.said() == []
    cid = live.session.conversation_id
    assert await _rule_decisions(app_engine, cid, "wrong_number_claim") == ["blocked"]


async def test_v3_stated_wrong_number_ends_the_call(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    live, _, _, rec = await _live(api_client, make_debtor, app_engine, "v3")
    await live.debtor_says("I'm sorry, I don’t know anyone by that name.")
    result, _ = await live.handlers["report_third_party"]({"kind": "wrong_number"}, live.flow)
    assert result["status"] == "ended"
    assert live.session.outcome == Outcome.END_WRONG_PERSON.value
    assert rec.said() == [scripts.WRONG_NUMBER_END]
    cid = live.session.conversation_id
    assert await _rule_decisions(app_engine, cid, "wrong_number_claim") == ["allowed"]


async def test_v3_not_available_is_unchanged(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    live, _, _, _ = await _live(api_client, make_debtor, app_engine, "v3")
    await live.handlers["report_third_party"]({"kind": "not_available"}, live.flow)
    assert live.session.outcome == Outcome.THIRD_PARTY_CALLBACK.value


# --- v3: "already paid" -> dispute handoff by code (fixes.payment_claim_handoff) ------------


async def _verified_live(
    client: httpx.AsyncClient, make_debtor: MakeDebtor, engine: AsyncEngine, version: str
) -> tuple[LiveChat, StubWorker, Recorder]:
    live, worker, _, rec = await _live(client, make_debtor, engine, version)
    await live.debtor_says(f"{NAME}, 1993-08-08")  # v2/v3: verified by code
    assert live.session.node == "negotiate"
    return live, worker, rec


async def _dispute_reason(engine: AsyncEngine, cid: int) -> str | None:
    async with engine.connect() as conn:
        debtor_id = await conn.scalar(
            select(Event.debtor_id).where(Event.conversation_id == cid).limit(1)
        )
        return await conn.scalar(select(Dispute.reason).where(Dispute.debtor_id == debtor_id))


async def test_v3_payment_claim_hands_off_as_dispute(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    live, worker, rec = await _verified_live(api_client, make_debtor, app_engine, "v3")
    await live.debtor_says("I paid this in full last March. I owe nothing.")
    assert live.session.outcome == Outcome.HANDOFF.value
    assert rec.said()[-1] == scripts.HANDOFF_END
    cid = live.session.conversation_id
    assert await _rule_decisions(app_engine, cid, "payment_claim") == ["blocked"]
    # Same context shape as an LLM tool call (D-006), and the LLM is not run.
    assert worker.appended()[-3:] == [("user", False), ("assistant", False), ("tool", False)]
    reason = await _dispute_reason(app_engine, cid)
    assert reason is not None and "payment claim" in reason


async def test_v3_no_payment_claim_lets_the_llm_answer(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    live, worker, _ = await _verified_live(api_client, make_debtor, app_engine, "v3")
    await live.debtor_says("I can pay half now.")
    assert not live.session.ended.is_set()
    assert worker.appended()[-1] == ("user", True)
    cid = live.session.conversation_id
    assert await _rule_decisions(app_engine, cid, "payment_claim") == ["allowed"]


async def test_v3_payment_claim_before_verification_is_not_checked(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    live, _, _, _ = await _live(api_client, make_debtor, app_engine, "v3")
    await live.debtor_says("I already paid, stop calling.")
    assert not live.session.ended.is_set()
    cid = live.session.conversation_id
    assert await _rule_decisions(app_engine, cid, "payment_claim") == []


async def test_v2_payment_claim_is_left_to_the_llm(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor, app_engine: AsyncEngine
) -> None:
    live, worker, _ = await _verified_live(api_client, make_debtor, app_engine, "v2")
    await live.debtor_says("I paid this in full last March.")
    assert not live.session.ended.is_set() and worker.appended()[-1] == ("user", True)
    cid = live.session.conversation_id
    assert await _rule_decisions(app_engine, cid, "payment_claim") == []


@pytest.mark.parametrize(
    ("version", "code", "instruction"),
    [("v3", "invalid_offer", True), ("v2", "discount_not_allowed", False)],
)
async def test_zero_discount_offer_by_version(
    api_client: httpx.AsyncClient,
    make_debtor: MakeDebtor,
    app_engine: AsyncEngine,
    version: str,
    code: str,
    instruction: bool,
) -> None:
    live, _, _ = await _verified_live(api_client, make_debtor, app_engine, version)
    out, nxt = await live.handlers["check_discount_request"]({"offered_total_jpy": 0}, live.flow)
    assert out["approved"] is False and nxt is None
    assert ("request_handoff" in out.get("instruction", "")) is instruction
    async with app_engine.connect() as conn:
        codes = await conn.scalar(
            select(Event.input["codes"]).where(
                Event.conversation_id == live.session.conversation_id,
                Event.rule == "no_discount",
            )
        )
    assert codes == [code]
