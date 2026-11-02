"""Release gate against a committed baseline summary (no DB, no LLM)."""

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from evals.compare import ComparisonError
from evals.gate import BASELINE_DIR, baseline_doc, evaluate, read_baseline, render
from tests.test_compare import SIMULATED, _run, _scores

EXPORTED = datetime(2026, 10, 25, tzinfo=UTC)
BASE_RUNS = [_run(1, "v1"), _run(2, "v1", "adversarial", simulator_version="sim-v1")]


def _baseline(**score_overrides: Any) -> dict[str, Any]:
    scores = (
        _scores(1, "base/a", **score_overrides)
        + _scores(1, "base/b")
        + _scores(2, "adversarial/x", near=1)
    )
    # Round-trip through JSON, exactly like the committed file.
    doc: dict[str, Any] = json.loads(json.dumps(baseline_doc(BASE_RUNS, scores, EXPORTED)))
    return doc


def _candidate(**score_overrides: Any) -> tuple[list[Any], list[Any]]:
    runs = [_run(10, "v2"), _run(11, "v2", "adversarial")]
    scores = (
        _scores(10, "base/a", **score_overrides)
        + _scores(10, "base/b")
        + _scores(11, "adversarial/x", near=1)
    )
    return runs, scores


def test_baseline_round_trips_cells_and_fingerprints() -> None:
    doc = _baseline()
    assert doc["label"] == "SYNTHETIC" and doc["agent_version"] == "v1"
    assert [r["id"] for r in doc["runs"]] == [1, 2]
    assert doc["runs"][0]["config_hash"] == "config-v1"
    assert any("config hash" in n.lower() for n in doc["notes"])
    _, cells = read_baseline(doc)
    assert cells[("base", "base/a")].outcome_pass == 1
    assert cells[("adversarial", "adversarial/x")].run_ids == (2,)


def test_baseline_refuses_mixed_versions() -> None:
    with pytest.raises(ComparisonError, match="mix agent versions"):
        baseline_doc([_run(1, "v1"), _run(2, "v2")], [], EXPORTED)


def test_equal_results_pass() -> None:
    result = evaluate(_baseline(), *_candidate(), SIMULATED)
    assert result.verdict == "PASS" and result.failures == result.warnings == []


def test_any_critical_fails() -> None:
    result = evaluate(_baseline(), *_candidate(critical=1), SIMULATED)
    assert result.verdict == "FAIL"
    assert result.failures == ["`base/a`: 1 critical violation(s)"]


def test_outcome_drop_fails_but_a_gain_does_not() -> None:
    dropped = evaluate(_baseline(), *_candidate(outcome=False), SIMULATED)
    assert dropped.failures == ["`base/a`: outcome match dropped 1/1 → 0/1"]
    gained = evaluate(_baseline(outcome=False), *_candidate(), SIMULATED)
    assert gained.verdict == "PASS"


def test_judge_drop_over_threshold_is_only_a_warning() -> None:
    small = evaluate(_baseline(judge=4.0), *_candidate(judge=3.0), SIMULATED)
    # base/a drops 1.0 but the average over 3 scenarios drops 0.33 > 0.3.
    assert small.verdict == "WARNING" and small.failures == []
    assert all("average" in w for w in small.warnings)
    within = evaluate(_baseline(judge=4.0), *_candidate(judge=3.5), SIMULATED)
    assert within.verdict == "PASS"


def test_near_miss_rise_is_a_warning() -> None:
    result = evaluate(_baseline(), *_candidate(near=2), SIMULATED)
    assert result.verdict == "WARNING"
    assert result.warnings == ["near-miss rate 8.3% → 25.0%"]


def test_missing_scenario_in_a_suite_that_ran_fails() -> None:
    runs, scores = _candidate()
    scores = [s for s in scores if s.scenario_id != "base/b"]
    assert evaluate(_baseline(), runs, scores, SIMULATED).failures == [
        "`base/b`: no result (incomplete gate)"
    ]


def test_infra_error_counts_as_missing() -> None:
    runs, scores = _candidate()
    scores = [s for s in scores if s.scenario_id != "base/b"] + _scores(10, "base/b", infra=True)
    result = evaluate(_baseline(), runs, scores, SIMULATED)
    assert result.failures == ["`base/b`: no result (incomplete gate)"]


def test_suite_not_run_is_a_note_not_a_failure() -> None:
    runs, scores = _candidate()
    result = evaluate(_baseline(), runs[:1], [s for s in scores if s.eval_run_id == 10], SIMULATED)
    assert result.verdict == "PASS"
    assert "suite `adversarial` not run" in result.notes


def test_refuses_incomparable_runs() -> None:
    runs, scores = _candidate()
    runs[0] = _run(10, "v2", rubric_version="rubric-v3")
    with pytest.raises(ComparisonError, match="rubric_version"):
        evaluate(_baseline(), runs, scores, SIMULATED)


def test_report_is_labeled_and_lists_failures() -> None:
    text = render(evaluate(_baseline(), *_candidate(critical=1), SIMULATED))
    assert text.startswith("# dueflow release gate: **FAIL**")
    assert "**Label: SYNTHETIC**" in text
    assert "- `base/a`: 1 critical violation(s)" in text


def test_committed_v1_baseline_is_valid() -> None:
    doc = json.loads((BASELINE_DIR / "v1.json").read_text(encoding="utf-8"))
    runs, cells = read_baseline(doc)
    assert doc["label"] == "SYNTHETIC" and doc["agent_version"] == "v1"
    assert sorted(r.id for r in runs) == [5, 7, 8, 9]
    assert len(cells) == 44 and doc["not_included"] == []


def test_aborted_runs_are_never_gated_or_exported() -> None:
    runs, scores = _candidate()
    runs[0] = _run(10, "v2", status="aborted", status_reason="D-006")
    with pytest.raises(ComparisonError, match="aborted"):
        evaluate(_baseline(), runs, scores, SIMULATED)
    with pytest.raises(ComparisonError, match="aborted"):
        baseline_doc([_run(1, "v1", status="aborted")], [], EXPORTED)


def test_gate_rules_change_needs_the_flag_and_is_noted() -> None:
    runs, scores = _candidate()
    runs = [
        _run(10, "v3", rules_version="rules-v7"),
        _run(11, "v3", "adversarial", rules_version="rules-v7"),
    ]
    with pytest.raises(ComparisonError, match="rules_version"):
        evaluate(_baseline(), runs, scores, SIMULATED)
    result = evaluate(_baseline(), runs, scores, SIMULATED, ("rules-v6", "rules-v7"))
    assert result.verdict == "PASS"
    assert any("`rules-v6` → `rules-v7`" in n for n in result.notes)
