"""Loads rules.yaml once at startup.

Rule functions receive the parsed values; they never read files themselves.
"""

from datetime import time
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_RULES_PATH = Path(__file__).with_name("rules.yaml")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ContactWindow(_Frozen):
    start: time
    end: time

    @model_validator(mode="after")
    def _start_before_end(self) -> Self:
        if self.start >= self.end:
            raise ValueError("contact_window.start must be before end")
        return self


class AttemptLimits(_Frozen):
    per_local_day: int = Field(ge=1)
    per_rolling_7_days: int = Field(ge=1)


class PlanLimits(_Frozen):
    min_months: int = Field(ge=1)
    max_months: int = Field(ge=1)
    min_down_payment_pct: int = Field(ge=0, le=100)
    min_down_payment_pct_after_broken_promise: int = Field(ge=0, le=100)
    down_payment_due_days: int = Field(ge=0)

    @model_validator(mode="after")
    def _min_le_max(self) -> Self:
        if self.min_months > self.max_months:
            raise ValueError("plan.min_months must be <= max_months")
        return self


class PayInFullLimits(_Frozen):
    max_days_until_payment: int = Field(ge=0)


class IdentityLimits(_Frozen):
    max_failed_attempts: int = Field(ge=1)
    max_turns_without_attempt: int = Field(ge=1)


class OutputCheckLists(_Frozen):
    debt_words: tuple[str, ...] = Field(min_length=1)
    banned_phrases: tuple[str, ...] = Field(min_length=1)
    relative_date_words: tuple[str, ...] = Field(min_length=1)


class ThirdPartyLists(_Frozen):
    wrong_number_phrases: tuple[str, ...] = Field(min_length=1)


class DisputeLists(_Frozen):
    payment_claim_phrases: tuple[str, ...] = Field(min_length=1)


class LadderLimits(_Frozen):
    """rules-v8 (agent v5): how far the engine's payment ladder may go in one call."""

    max_counter_offers: int = Field(ge=0)


class CallbackLimits(_Frozen):
    """rules-v8 (agent v5): how far ahead a customer-requested callback may be scheduled."""

    max_days_ahead: int = Field(ge=1)


class ClaimLists(_Frozen):
    """rules-v8 (agent v5): statements the agent has no data for, never said."""

    unverified_phrases: tuple[str, ...] = Field(min_length=1)


class PolicyRules(_Frozen):
    version: str
    contact_window: ContactWindow
    attempt_limits: AttemptLimits
    plan: PlanLimits
    pay_in_full: PayInFullLimits
    identity: IdentityLimits
    output_check: OutputCheckLists
    # rules-v7. Optional so rules-v6 still loads (tests compare v1/v2 under both).
    third_party: ThirdPartyLists | None = None
    dispute: DisputeLists | None = None
    # rules-v8 (agent v5 only). Optional so rules-v7 still loads (tests compare v1-v4 under both).
    ladder: LadderLimits | None = None
    callback: CallbackLimits | None = None
    claims: ClaimLists | None = None


def load_rules(path: Path = DEFAULT_RULES_PATH) -> PolicyRules:
    return PolicyRules.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
