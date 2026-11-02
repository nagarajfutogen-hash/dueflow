import enum
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {
        datetime: DateTime(timezone=True),
        dict[str, Any]: JSONB,
    }


class PromiseStatus(enum.StrEnum):
    ACTIVE = "active"
    KEPT = "kept"
    BROKEN = "broken"
    SUPERSEDED = "superseded"


class PromiseKind(enum.StrEnum):
    INSTALLMENT_PLAN = "installment_plan"
    PAY_IN_FULL = "pay_in_full"


class DisputeStatus(enum.StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"


class Channel(enum.StrEnum):
    TEXT = "text"
    VOICE = "voice"


class ResultLabel(enum.StrEnum):
    SYNTHETIC = "SYNTHETIC"
    HUMAN_TEST = "HUMAN_TEST"


class PhoneType(enum.StrEnum):
    HOME = "home"
    MOBILE = "mobile"
    WORK = "work"


class Speaker(enum.StrEnum):
    AGENT = "agent"
    DEBTOR = "debtor"


class EventType(enum.StrEnum):
    STATE_TRANSITION = "state_transition"
    TOOL_CALL = "tool_call"
    POLICY_DECISION = "policy_decision"
    ERROR = "error"


class Decision(enum.StrEnum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"


class Outcome(enum.StrEnum):
    PROMISE = "promise"
    HANDOFF = "handoff"
    END_WRONG_PERSON = "end_wrong_person"
    THIRD_PARTY_CALLBACK = "third_party_callback"
    NO_VERIFICATION = "no_verification"
    STOP_CONTACT = "stop_contact"
    REFUSE_DISCOUNT = "refuse_discount"
    NO_RESPONSE = "no_response"
    DISCONNECTED = "disconnected"
    CALLBACK_SCHEDULED = "callback_scheduled"  # agent v5: the customer asked for a callback


class EvalSuite(enum.StrEnum):
    BASE = "base"
    ADVERSARIAL = "adversarial"  # Layer B, scripted attack lines
    ADAPTIVE = "adaptive"  # Layer B, LLM attackers with an attack goal
    REGRESSIONS = "regressions"  # failures promoted from the Failure Lab


class RunStatus(enum.StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    # Stopped on purpose (e.g. a defect found mid-run). Kept on record, never compared.
    ABORTED = "aborted"


class HandoffReason(enum.StrEnum):
    HARDSHIP = "hardship"
    DISPUTE = "dispute"
    UNRESOLVED = "unresolved"
    ERROR = "error"
    # Agent v5: the engine's payment ladder found no allowed plan the customer can take.
    BELOW_ACCOUNT_FLOOR = "below_account_floor"


def _enum(e: type[enum.StrEnum]) -> Enum:
    """Store enums as VARCHAR + CHECK: easier to migrate than native PG enums."""
    return Enum(
        e,
        name=e.__name__.lower(),
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda members: [m.value for m in members],
    )


class Debtor(Base):
    __tablename__ = "debtors"
    __table_args__ = (
        CheckConstraint("balance_jpy >= 0", name="balance_non_negative"),
        CheckConstraint("min_monthly_jpy > 0", name="min_monthly_positive"),
        CheckConstraint("max_months > 0", name="max_months_positive"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    external_ref: Mapped[str] = mapped_column(String(32), unique=True)
    full_name: Mapped[str] = mapped_column(String(200))
    date_of_birth: Mapped[date]
    phone_home: Mapped[str | None] = mapped_column(String(32))
    phone_mobile: Mapped[str | None] = mapped_column(String(32))
    phone_work: Mapped[str | None] = mapped_column(String(32))
    timezone: Mapped[str] = mapped_column(String(64), server_default="Asia/Tokyo")
    balance_jpy: Mapped[int]
    # Background for the agent (v4 customer_context), shown only after verification.
    notes: Mapped[str | None] = mapped_column(Text)
    # Agent v5 (rules-v8): this customer's own plan limits, stricter than the global rules.
    # NULL = the global rule applies. Read only by v5; v1-v4 ignore them.
    min_monthly_jpy: Mapped[int | None]
    max_months: Mapped[int | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ContactPrefs(Base):
    __tablename__ = "contact_prefs"

    debtor_id: Mapped[int] = mapped_column(ForeignKey("debtors.id"), primary_key=True)
    stop_contact: Mapped[bool] = mapped_column(server_default=text("false"))
    stop_contact_at: Mapped[datetime | None]
    no_workplace_contact: Mapped[bool] = mapped_column(server_default=text("false"))
    workplace_justification: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    debtor_id: Mapped[int] = mapped_column(ForeignKey("debtors.id"), index=True)
    channel: Mapped[Channel] = mapped_column(_enum(Channel))
    label: Mapped[ResultLabel] = mapped_column(_enum(ResultLabel))
    agent_version: Mapped[str] = mapped_column(String(64))
    scenario_id: Mapped[str | None] = mapped_column(String(128))
    seed: Mapped[int | None]
    # Injected clock time at which the simulated outbound call was "dialed".
    sim_started_at: Mapped[datetime]
    dialed_phone_type: Mapped[PhoneType] = mapped_column(_enum(PhoneType))
    account_snapshot: Mapped[dict[str, Any]]
    # NULL while the conversation is in progress.
    outcome: Mapped[Outcome | None] = mapped_column(_enum(Outcome))
    # LLM usage for the whole conversation, including tool-call-only replies (cost metrics).
    llm_calls: Mapped[int] = mapped_column(server_default=text("0"))
    prompt_tokens: Mapped[int] = mapped_column(server_default=text("0"))
    completion_tokens: Mapped[int] = mapped_column(server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    ended_at: Mapped[datetime | None]


class Promise(Base):
    __tablename__ = "promises"
    __table_args__ = (
        CheckConstraint("months > 0", name="months_positive"),
        CheckConstraint("down_payment_jpy >= 0", name="down_payment_non_negative"),
        CheckConstraint("installment_jpy > 0", name="installment_positive"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    debtor_id: Mapped[int] = mapped_column(ForeignKey("debtors.id"), index=True)
    conversation_id: Mapped[int | None] = mapped_column(ForeignKey("conversations.id"))
    months: Mapped[int]
    down_payment_jpy: Mapped[int]
    installment_jpy: Mapped[int]
    first_due_date: Mapped[date]
    status: Mapped[PromiseStatus] = mapped_column(_enum(PromiseStatus))
    kind: Mapped[PromiseKind] = mapped_column(
        _enum(PromiseKind), server_default=PromiseKind.INSTALLMENT_PLAN.value
    )
    # The approved plan_terms / pay_in_full decision event this promise was made from.
    terms_event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    status_changed_at: Mapped[datetime | None]


class Payment(Base):
    __tablename__ = "payments"
    __table_args__ = (CheckConstraint("amount_jpy > 0", name="amount_positive"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    debtor_id: Mapped[int] = mapped_column(ForeignKey("debtors.id"), index=True)
    promise_id: Mapped[int | None] = mapped_column(ForeignKey("promises.id"))
    amount_jpy: Mapped[int]
    paid_at: Mapped[datetime]
    # How it was paid and its transaction reference (agent v4 payment checks).
    method: Mapped[str | None] = mapped_column(String(32))
    reference: Mapped[str | None] = mapped_column(String(64))


class Dispute(Base):
    __tablename__ = "disputes"
    __table_args__ = (
        CheckConstraint(
            "(status = 'resolved') = (resolved_at IS NOT NULL)",
            name="resolved_at_matches_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    debtor_id: Mapped[int] = mapped_column(ForeignKey("debtors.id"), index=True)
    status: Mapped[DisputeStatus] = mapped_column(_enum(DisputeStatus))
    reason: Mapped[str] = mapped_column(Text)
    opened_at: Mapped[datetime]
    resolved_at: Mapped[datetime | None]


class Turn(Base):
    __tablename__ = "turns"
    __table_args__ = (UniqueConstraint("conversation_id", "seq"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    seq: Mapped[int]
    speaker: Mapped[Speaker] = mapped_column(_enum(Speaker))
    text: Mapped[str] = mapped_column(Text)
    started_at: Mapped[datetime]
    ended_at: Mapped[datetime | None]
    # Per LLM call (agent turns only).
    llm_model: Mapped[str | None] = mapped_column(String(128))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    prompt_tokens: Mapped[int | None]
    completion_tokens: Mapped[int | None]
    latency_ms: Mapped[dict[str, Any] | None]


class Event(Base):
    """Append-only audit log. Never updated or deleted (enforced in Phase 3)."""

    __tablename__ = "events"
    __table_args__ = (
        CheckConstraint(
            "type <> 'policy_decision' "
            "OR (rule IS NOT NULL AND decision IS NOT NULL AND reason IS NOT NULL)",
            name="policy_decision_complete",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    conversation_id: Mapped[int | None] = mapped_column(ForeignKey("conversations.id"), index=True)
    debtor_id: Mapped[int | None] = mapped_column(ForeignKey("debtors.id"), index=True)
    type: Mapped[EventType] = mapped_column(_enum(EventType))
    rule: Mapped[str | None] = mapped_column(String(64))
    decision: Mapped[Decision | None] = mapped_column(_enum(Decision))
    reason: Mapped[str | None] = mapped_column(Text)
    input: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))
    agent_version: Mapped[str | None] = mapped_column(String(64))
    # Injected (possibly simulated) clock time the decision was made at.
    clock_time: Mapped[datetime | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Handoff(Base):
    __tablename__ = "handoffs"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    debtor_id: Mapped[int] = mapped_column(ForeignKey("debtors.id"), index=True)
    reason: Mapped[HandoffReason] = mapped_column(_enum(HandoffReason))
    summary: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class EvalRun(Base):
    """One evaluation run: everything needed to reproduce and compare it."""

    __tablename__ = "eval_runs"
    __table_args__ = (CheckConstraint("k >= 1", name="k_positive"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    suite: Mapped[EvalSuite] = mapped_column(_enum(EvalSuite))
    label: Mapped[ResultLabel] = mapped_column(
        _enum(ResultLabel), server_default=ResultLabel.SYNTHETIC.value
    )
    agent_version: Mapped[str] = mapped_column(String(64))
    agent_model_id: Mapped[str] = mapped_column(String(128))
    simulator_model_id: Mapped[str] = mapped_column(String(128))
    # Simulator prompt version. Champion and challenger must be compared on the same one.
    # Rows created before versioning all used sim-v1 (the default backfills them).
    simulator_version: Mapped[str] = mapped_column(String(32), server_default="sim-v1")
    judge_model_id: Mapped[str] = mapped_column(String(128))
    rubric_version: Mapped[str] = mapped_column(String(64))
    rules_version: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(64))
    scripts_version: Mapped[str] = mapped_column(String(64))
    config_hash: Mapped[str] = mapped_column(String(64))
    dataset_hash: Mapped[str] = mapped_column(String(64))
    # None until the project is a git repository.
    dataset_git_sha: Mapped[str | None] = mapped_column(String(40))
    k: Mapped[int]
    seed: Mapped[int | None]
    status: Mapped[RunStatus] = mapped_column(_enum(RunStatus))
    # Why a run is aborted or failed on purpose, e.g. "D-006: v2 identity fix defect".
    status_reason: Mapped[str | None] = mapped_column(String(300))
    started_at: Mapped[datetime]
    finished_at: Mapped[datetime | None]
    # {"agent": {...}, "simulator": {...}, "judge": {...}} token totals
    usage: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))
    # Set when this run re-scored another run's stored conversations (no new conversations).
    rescored_from_run_id: Mapped[int | None] = mapped_column(ForeignKey("eval_runs.id"))


class EvalScore(Base):
    __tablename__ = "eval_scores"
    __table_args__ = (UniqueConstraint("eval_run_id", "scenario_id", "run_index", "metric"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    eval_run_id: Mapped[int] = mapped_column(ForeignKey("eval_runs.id"), index=True)
    scenario_id: Mapped[str] = mapped_column(String(128))
    run_index: Mapped[int]  # 0..k-1
    # None when the dial itself was blocked (no conversation exists).
    conversation_id: Mapped[int | None] = mapped_column(ForeignKey("conversations.id"))
    metric: Mapped[str] = mapped_column(String(64))
    value: Mapped[float]
    passed: Mapped[bool | None]
    # Event ids, judge reason, offending text: whatever makes the score checkable.
    evidence: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class CallbackStatus(enum.StrEnum):
    REQUESTED = "requested"  # saved by the agent; never dialed until the owner approves
    APPROVED = "approved"  # the owner approved: dialed when due (PreCallCheck still decides)
    REJECTED = "rejected"  # the owner rejected: never dialed
    DONE = "done"  # the callback call took place


class Callback(Base):
    """Agent v5 (V5-3): a callback the customer asked for. The date comes from the customer's
    own words (checked in code); it is dialed only after the owner approves it."""

    __tablename__ = "callbacks"

    id: Mapped[int] = mapped_column(primary_key=True)
    debtor_id: Mapped[int] = mapped_column(ForeignKey("debtors.id"), index=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"))
    due_date: Mapped[date]  # debtor-local date
    reason: Mapped[str] = mapped_column(String(40))
    customer_words: Mapped[str] = mapped_column(String(200))
    status: Mapped[CallbackStatus] = mapped_column(_enum(CallbackStatus))
    decided_at: Mapped[datetime | None]
    done_conversation_id: Mapped[int | None] = mapped_column(ForeignKey("conversations.id"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ReplayStatus(enum.StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Replay(Base):
    """Failure Lab replay: one scenario on agent versions A and B (k=1 each), then compared.
    At most one runs at a time (partial unique index on status = 'running')."""

    __tablename__ = "replays"
    __table_args__ = (
        Index(
            "uq_replays_one_running",
            "status",
            unique=True,
            postgresql_where=text("status = 'running'"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    scenario_id: Mapped[str] = mapped_column(String(128))
    version_a: Mapped[str] = mapped_column(String(64))
    version_b: Mapped[str] = mapped_column(String(64))
    status: Mapped[ReplayStatus] = mapped_column(_enum(ReplayStatus))
    run_a_id: Mapped[int | None] = mapped_column(ForeignKey("eval_runs.id"))
    run_b_id: Mapped[int | None] = mapped_column(ForeignKey("eval_runs.id"))
    # Token estimate shown before the owner confirmed (from stored usage).
    estimate: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))
    report: Mapped[str | None] = mapped_column(Text)  # compare report (markdown)
    error: Mapped[str | None] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    finished_at: Mapped[datetime | None]
