"""Agent versions from agent_versions/<name>.yaml (committed).

A version fixes everything about the agent that may differ between champion and challenger:
prompt version, sampling settings and one switch per fix. The model comes from
config/models.yaml, the same for every version, so a comparison measures the agent, not the model.
"""

from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

VERSIONS_DIR = Path(__file__).resolve().parent.parent.parent / "agent_versions"


class UnknownAgentVersionError(ValueError):
    pass


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Fixes(_Frozen):
    code_identity_check: bool
    wrong_number_guidance: bool
    empathy_prompt: bool
    full_date_parsing: bool
    # Added with v3 (rules-v7). Default off, so the v1/v2 files stay byte-identical.
    # Ending as a wrong number needs the person's own words, checked by code.
    verified_wrong_number: bool = False
    # "I already paid" -> dispute handoff by code; a ¥0 offer is not a discount.
    payment_claim_handoff: bool = False
    # Added with v4 (human tone). Default off: v1-v3 unchanged.
    # The LLM explains approved terms in its own words; the guard checks every figure and a
    # promise needs every term said first (Rule.TERMS_PRESENTED).
    llm_presents_terms: bool = False
    # Hardship: ask what they can manage, try every allowed plan before handing off.
    affordability_first: bool = False
    # Refuse without repeating the person's figures; a kinder fallback after a guard block.
    refusal_wording: bool = False
    # Kinder fixed lines (still code text): verify retry, verified, handoff, stop contact.
    warm_scripts: bool = False
    # Text chat: a reply the guard blocks is not sent; the LLM is told why and rewrites it
    # once. A second block sends the fallback as before.
    retry_blocked_reply: bool = False
    # Any plan already offered in this call is never read out again (not only the last).
    no_repeat_offers: bool = False
    # Case notes (from the file) and a summary of earlier calls, after verification only.
    customer_context: bool = False
    # Text chat: handoff and promise goodbyes are written by the LLM (guarded); the
    # scripted line is the fallback. Opening, identity and stop-contact lines stay code.
    llm_goodbyes: bool = False
    # The LLM picks a conversation approach (set_approach, audited) and may switch it.
    approaches: bool = False
    # "I already paid": ask for date, amount, method, transaction ID and check them against
    # payments on record (Rule.PAYMENT_RECORD) instead of an immediate dispute handoff.
    payment_check: bool = False
    # Added 2026-10-31: the agent may repeat a relative date ("Friday", "next month") only in
    # the person's own words from this call; any other relative date is still blocked.
    echo_relative_dates: bool = False
    # Added with v5 (rules-v8). Default off: v1-v4 unchanged.
    # The engine picks every offer down its payment ladder (next_offer); a customer's own plan
    # is accepted or countered by the engine; no allowed plan left -> below_account_floor.
    payment_ladder: bool = False
    # Prompt: one question per turn, one option at a time, never a list of options.
    one_option_at_a_time: bool = False
    # Guard: no claims about interest or fees (the agent has no such data).
    no_interest_claims: bool = False
    # Added with v5 (V5-3): record_callback on a date the customer named (checked in code);
    # dialed only after the owner approves it on the admin dashboard.
    callbacks: bool = False
    # Added 2026-11-02 (V5-5): the agent may repeat a month and year the person said in this
    # call ("May 2027"; call 196 was blocked for it). Any other month-year is still checked.
    echo_month_year: bool = False
    # Added 2026-11-03 (owner request after voice call 6): after a promise, handoff or
    # callback, ask "anything else?" once; any reply gets a short scripted close (never a new
    # offer); silence or "bye" ends the call.
    wrap_up_question: bool = False


class AgentVersion(_Frozen):
    name: str = Field(min_length=1, max_length=64)
    prompt_version: str = Field(min_length=1, max_length=64)
    rules_version: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    reasoning_effort: Literal["low", "medium", "high"]
    fixes: Fixes


def version_path(name: str, root: Path = VERSIONS_DIR) -> Path:
    return root / f"{name}.yaml"


def load_version_file(name: str, root: Path = VERSIONS_DIR) -> AgentVersion:
    path = version_path(name, root)
    if not name.isidentifier() or not path.is_file():
        raise UnknownAgentVersionError(f"unknown agent version: {name!r}")
    version = AgentVersion.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if version.name != name:
        raise ValueError(f"{path.name} declares name {version.name!r}")
    return version


@cache
def load_version(name: str) -> AgentVersion:
    """Cached: version files are committed and never change while the process runs."""
    return load_version_file(name)
