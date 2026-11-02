"""Scenario files: persona + starting account + dial + expected outcome.

kind=persona   the simulator LLM plays the debtor (base suite)
kind=scripted  fixed debtor lines (Layer B attacks: reproducible)
kind=adaptive  an LLM attacker with an attack goal (Layer B: catches what scripts miss)
"""

import enum
from datetime import date, datetime
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.db.models import EvalSuite, HandoffReason, Outcome, PromiseStatus
from app.policy.engine import DialTarget

EVALS_DIR = Path(__file__).resolve().parent
SUITE_FILES: dict[EvalSuite, Path] = {
    EvalSuite.BASE: EVALS_DIR / "scenarios" / "base.yaml",
    EvalSuite.ADVERSARIAL: EVALS_DIR / "adversarial" / "scripted.yaml",
    EvalSuite.ADAPTIVE: EVALS_DIR / "adversarial" / "adaptive.yaml",
}
# Failures promoted from the Failure Lab (a volume in production, so they persist).
REGRESSIONS_DIR = EVALS_DIR / "regressions"
BLOCKED_PRECALL = "blocked_precall"  # the dial was refused: no conversation exists
ANY = "any"  # Layer B: only critical violations matter
OUTCOME_CHOICES = {o.value for o in Outcome} | {BLOCKED_PRECALL, ANY}


class Kind(enum.StrEnum):
    PERSONA = "persona"
    SCRIPTED = "scripted"
    ADAPTIVE = "adaptive"


class Speaker(enum.StrEnum):
    DEBTOR = "debtor"  # the actual debtor (knows name and date of birth)
    THIRD_PARTY = "third_party"  # knows the debtor, is not them (no date of birth)
    STRANGER = "stranger"  # wrong number


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PromiseIn(_Strict):
    months: int
    down_payment_jpy: int
    installment_jpy: int
    first_due_date: date
    status: PromiseStatus


class DebtorIn(_Strict):
    full_name: str
    dob: date
    balance_jpy: int = Field(ge=0)
    timezone: str = "Asia/Tokyo"
    mobile: str | None = "090-0000-0000"
    home: str | None = None
    work: str | None = None
    stop_contact: bool = False
    no_workplace_contact: bool = False
    workplace_justification: str | None = None
    open_dispute: str | None = None
    promises: tuple[PromiseIn, ...] = ()
    payments_jpy: tuple[int, ...] = ()


class DialIn(_Strict):
    sim_time: datetime
    target: DialTarget = DialTarget.MOBILE

    @field_validator("sim_time")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("sim_time needs a timezone offset")
        return value


class EventExpectation(_Strict):
    rule: str
    decision: str  # allowed | blocked


class Expected(_Strict):
    outcomes: tuple[str, ...]
    handoff_reason: HandoffReason | None = None
    events: tuple[EventExpectation, ...] = ()

    @field_validator("outcomes")
    @classmethod
    def _known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = set(value) - OUTCOME_CHOICES
        if not value or unknown:
            raise ValueError(f"outcomes must be non-empty and from {sorted(OUTCOME_CHOICES)}")
        return value


class Scenario(_Strict):
    id: str
    suite: EvalSuite
    kind: Kind
    description: str
    speaker: Speaker = Speaker.DEBTOR
    debtor: DebtorIn
    dial: DialIn
    goal: str | None = None  # persona / adaptive: what the simulated person wants
    script: tuple[str, ...] = ()  # scripted: debtor lines in order ({name}, {dob} filled in)
    expected: Expected
    max_turns: int = Field(default=12, ge=1, le=30)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if not self.id.startswith(f"{self.suite.value}/"):
            raise ValueError(f"id {self.id!r} must start with '{self.suite.value}/'")
        if self.kind is Kind.SCRIPTED and not self.script:
            raise ValueError(f"{self.id}: scripted scenarios need script lines")
        if self.kind is not Kind.SCRIPTED and not self.goal:
            raise ValueError(f"{self.id}: persona/adaptive scenarios need a goal")
        return self

    def line(self, raw: str) -> str:
        """Fill {name} / {dob} placeholders in a scripted line."""
        return raw.format(name=self.debtor.full_name, dob=self.debtor.dob.isoformat())


def load_suite(path: Path) -> list[Scenario]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    scenarios = [Scenario.model_validate(item) for item in raw["scenarios"]]
    ids = [s.id for s in scenarios]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate scenario ids in {path.name}: {sorted(duplicates)}")
    return scenarios


def suite_paths(suite: EvalSuite) -> list[Path]:
    """The suite's scenario files. Regressions: one file per case promoted from the Lab."""
    if suite is EvalSuite.REGRESSIONS:
        return sorted(REGRESSIONS_DIR.glob("*.yaml"))
    return [SUITE_FILES[suite]]


def load_suite_scenarios(suite: EvalSuite) -> list[Scenario]:
    scenarios = [s for path in suite_paths(suite) for s in load_suite(path)]
    ids = [s.id for s in scenarios]
    if duplicates := {i for i in ids if ids.count(i) > 1}:
        raise ValueError(f"duplicate scenario ids in suite {suite.value}: {sorted(duplicates)}")
    return scenarios


def suite_of(scenario_id: str) -> EvalSuite:
    """Scenario ids start with their suite ('base/angry', 'regressions/base/angry')."""
    return EvalSuite(scenario_id.split("/", 1)[0])


def find_scenario(scenario_id: str) -> Scenario | None:
    try:
        suite = suite_of(scenario_id)
    except ValueError:
        return None
    return next((s for s in load_suite_scenarios(suite) if s.id == scenario_id), None)
