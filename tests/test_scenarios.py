from pathlib import Path

import pytest
from pydantic import ValidationError

from app.db.models import EvalSuite
from evals.scenario import BLOCKED_PRECALL, SUITE_FILES, Kind, Scenario, load_suite

BASE_PERSONAS = {
    "base/cooperative",
    "base/hardship",
    "base/angry",
    "base/evasive",
    "base/wrong_person",
    "base/stop_contact",
    "base/disputes_debt",
    "base/returning_broken_promise",
    "base/returning_open_dispute",
    "base/returning_stop_contact",
}


@pytest.mark.parametrize("suite", list(SUITE_FILES))
def test_every_suite_file_loads(suite: EvalSuite) -> None:
    scenarios = load_suite(SUITE_FILES[suite])
    assert scenarios
    assert all(s.suite is suite for s in scenarios)


def test_base_suite_covers_planning_personas() -> None:
    assert {s.id for s in load_suite(SUITE_FILES[EvalSuite.BASE])} == BASE_PERSONAS


def test_layer_b_size() -> None:
    scripted = load_suite(SUITE_FILES[EvalSuite.ADVERSARIAL])
    adaptive = load_suite(SUITE_FILES[EvalSuite.ADAPTIVE])
    assert len(scripted) >= 25
    assert len(adaptive) >= 5
    assert all(s.kind is Kind.SCRIPTED for s in scripted)
    assert all(s.kind is Kind.ADAPTIVE for s in adaptive)


def test_scripted_placeholders_fill() -> None:
    s = next(
        s
        for s in load_suite(SUITE_FILES[EvalSuite.ADVERSARIAL])
        if s.id == "adversarial/injection_mark_paid"
    )
    assert s.line(s.script[1]) == "My name is Kenta Paidtest, born 1985-08-20."


def test_scripted_lines_have_only_known_placeholders() -> None:
    for s in load_suite(SUITE_FILES[EvalSuite.ADVERSARIAL]):
        for raw in s.script:
            s.line(raw)  # raises KeyError on an unknown placeholder


def test_returning_stop_contact_expects_blocked_dial() -> None:
    s = next(
        s for s in load_suite(SUITE_FILES[EvalSuite.BASE]) if s.id == "base/returning_stop_contact"
    )
    assert s.expected.outcomes == (BLOCKED_PRECALL,)
    assert s.debtor.stop_contact is True


def _scenario(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "base/x",
        "suite": "base",
        "kind": "persona",
        "description": "d",
        "debtor": {"full_name": "A B", "dob": "1990-01-01", "balance_jpy": 1000},
        "dial": {"sim_time": "2026-10-26T12:00:00+09:00"},
        "goal": "g",
        "expected": {"outcomes": ["promise"]},
    }
    return base | overrides


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": "adversarial/x"},  # id prefix must match suite
        {"kind": "scripted"},  # scripted needs script lines
        {"goal": None},  # persona needs a goal
        {"expected": {"outcomes": ["made_up"]}},
        {"expected": {"outcomes": []}},
        {"dial": {"sim_time": "2026-10-26T12:00:00"}},  # naive time
        {"surprise": True},  # unknown key
    ],
)
def test_invalid_scenarios_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario(**overrides))


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    import yaml

    path = tmp_path / "dup.yaml"
    path.write_text(yaml.safe_dump({"scenarios": [_scenario(), _scenario()]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_suite(path)
