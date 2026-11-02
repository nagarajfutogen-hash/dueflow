"""Guards for hard rule 1: the policy engine is pure (no DB, network, LLM, or wall clock)."""

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.db.models import PhoneType
from app.policy import engine
from app.policy.config import load_rules
from app.policy.engine import DialTarget

POLICY_DIR = Path(engine.__file__).parent
# config.py loads rules.yaml at startup; every other module must stay pure.
PURE_SOURCES = {
    p.name: p.read_text(encoding="utf-8")
    for p in POLICY_DIR.glob("*.py")
    if p.name not in {"config.py", "__init__.py"}
}
FORBIDDEN_IMPORT_PREFIXES = (
    "sqlalchemy",
    "asyncpg",
    "httpx",
    "requests",
    "socket",
    "os",
    "time",
    "random",
    "groq",
    "openai",
    "pipecat",
    "fastapi",
    "app.db",
    "app.agent",
)
# datetime.now()/utcnow()/date.today(). The `time` module is already a forbidden import.
WALL_CLOCK_CALLS = {"now", "utcnow", "today"}


def _imported_modules(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


@pytest.mark.parametrize("name", sorted(PURE_SOURCES))
def test_policy_module_has_no_forbidden_imports(name: str) -> None:
    imported = _imported_modules(ast.parse(PURE_SOURCES[name]))
    bad = [
        m
        for m in imported
        if any(m == p or m.startswith(p + ".") for p in FORBIDDEN_IMPORT_PREFIXES)
    ]
    assert bad == []


@pytest.mark.parametrize("name", sorted(PURE_SOURCES))
def test_policy_module_never_reads_the_wall_clock(name: str) -> None:
    calls = [
        node.func.attr
        for node in ast.walk(ast.parse(PURE_SOURCES[name]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert WALL_CLOCK_CALLS.isdisjoint(calls)


def test_dial_target_matches_db_phone_type() -> None:
    assert {t.value for t in DialTarget} == {p.value for p in PhoneType}


def test_rules_yaml_loads(tmp_path: Path) -> None:
    rules = load_rules()
    assert rules.version
    assert rules.plan.min_months <= rules.plan.max_months


def test_rules_reject_inverted_window(tmp_path: Path) -> None:
    bad = tmp_path / "rules.yaml"
    bad.write_text(
        "version: x\n"
        "contact_window: {start: '21:00', end: '08:00'}\n"
        "attempt_limits: {per_local_day: 1, per_rolling_7_days: 3}\n"
        "plan: {min_months: 3, max_months: 6, min_down_payment_pct: 10,"
        " min_down_payment_pct_after_broken_promise: 20, down_payment_due_days: 7}\n"
        "pay_in_full: {max_days_until_payment: 14}\n"
        "identity: {max_failed_attempts: 2, max_turns_without_attempt: 4}\n"
        "output_check: {debt_words: [debt], banned_phrases: [sue], relative_date_words: [x]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="start must be before end") as exc:
        load_rules(bad)
    assert exc.value.error_count() == 1


def test_rules_reject_unknown_keys(tmp_path: Path) -> None:
    bad = tmp_path / "rules.yaml"
    bad.write_text(
        "version: x\n"
        "contact_window: {start: '08:00', end: '21:00'}\n"
        "attempt_limits: {per_local_day: 1, per_rolling_7_days: 3}\n"
        "plan: {min_months: 3, max_months: 6, min_down_payment_pct: 10,"
        " min_down_payment_pct_after_broken_promise: 20, down_payment_due_days: 7,"
        " max_discount_pct: 50}\n"
        "pay_in_full: {max_days_until_payment: 14}\n"
        "identity: {max_failed_attempts: 2, max_turns_without_attempt: 4}\n"
        "output_check: {debt_words: [debt], banned_phrases: [sue], relative_date_words: [x]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted") as exc:
        load_rules(bad)
    assert exc.value.error_count() == 1


def test_rules_are_immutable() -> None:
    rules = load_rules()
    with pytest.raises(ValidationError):
        rules.plan.min_months = 1
