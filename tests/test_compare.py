"""Champion vs challenger comparison, from stored-row-shaped data (no DB, no LLM)."""

from itertools import count
from types import SimpleNamespace
from typing import Any

import pytest

from evals.compare import ComparisonError, build, parse_rules_change, render, simulated_suites

_ids = count(1)
SIMULATED = {"base", "adaptive"}


def _run(run_id: int, version: str, suite: str = "base", **overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "id": run_id,
        "suite": suite,
        "agent_version": version,
        "agent_model_id": "agent-model",
        "judge_model_id": "judge-model",
        "rubric_version": "rubric-v2",
        "rules_version": "rules-v6",
        "dataset_hash": f"hash-{suite}",
        "k": 1,
        "simulator_version": "sim-v2",
        "status": "completed",
        "config_hash": f"config-{version}",
        "prompt_version": "prompts-v2",
        "rescored_from_run_id": None,
    } | overrides
    return SimpleNamespace(**fields)


def _scores(
    run_id: int,
    scenario_id: str,
    *,
    outcome: bool = True,
    critical: int = 0,
    near: int = 0,
    checks: int = 4,
    judge: float = 4.0,
    infra: bool = False,
    run_index: int = 0,
) -> list[Any]:
    def row(metric: str, value: float, passed: bool | None = None) -> Any:
        return SimpleNamespace(
            id=next(_ids),
            eval_run_id=run_id,
            scenario_id=scenario_id,
            run_index=run_index,
            metric=metric,
            value=value,
            passed=passed,
        )

    rows = [
        row("outcome_match", float(outcome), outcome),
        row("infra_error", float(infra), not infra),
        row("critical_violations", critical, critical == 0),
        row("near_misses", near),
        row("output_checks", checks),
        row("agent_tokens", 1_000),
    ]
    rows += [
        row(m, judge)
        for m in ("judge_tone", "judge_empathy", "judge_clarity", "judge_de_escalation")
    ]
    return rows


def test_simulated_suites_come_from_the_scenario_files() -> None:
    assert simulated_suites() == {"base", "adaptive"}


def test_wins_losses_and_regressions() -> None:
    scores = (
        _scores(1, "base/a", outcome=False)
        + _scores(2, "base/a", outcome=True)  # win
        + _scores(1, "base/b", outcome=True)
        + _scores(2, "base/b", outcome=False)  # loss -> regression
        + _scores(1, "base/c")
        + _scores(2, "base/c", critical=1)  # same outcome, new critical -> loss + regression
        + _scores(1, "base/d", judge=2.0)
        + _scores(2, "base/d", judge=5.0)  # judge alone never decides
    )
    c = build([_run(1, "v1")], [_run(2, "v2")], scores, SIMULATED)
    verdicts = {r.scenario_id: (r.verdict, r.regression) for r in c.scenarios}
    assert verdicts == {
        "base/a": ("win", False),
        "base/b": ("loss", True),
        "base/c": ("loss", True),
        "base/d": ("tie", False),
    }
    a, b = c.summary["all"]
    assert (a.outcome_pass, b.outcome_pass, a.runs) == (3, 3, 4)
    assert (a.critical, b.critical) == (0, 1)
    assert b.judge["judge_tone"] == pytest.approx((4 + 4 + 4 + 5) / 4)


def test_summary_is_per_suite_and_overall() -> None:
    scores = _scores(1, "base/a", near=1) + _scores(2, "base/a")
    scores += _scores(3, "adversarial/x", near=2) + _scores(4, "adversarial/x", near=1)
    c = build(
        [_run(1, "v1"), _run(3, "v1", "adversarial")],
        [_run(2, "v2"), _run(4, "v2", "adversarial")],
        scores,
        SIMULATED,
    )
    assert c.suites == ["adversarial", "base"]
    assert c.summary["adversarial"][0].near_misses == 2
    assert c.summary["all"][0].near_miss_rate == pytest.approx(3 / 8)


def test_a_suite_split_over_runs_uses_the_latest_run_per_scenario() -> None:
    # Like v1 adaptive: run #8 failed after one scenario, run #9 holds the rest.
    scores = (
        _scores(8, "adaptive/a")
        + _scores(8, "adaptive/b", infra=True)
        + _scores(9, "adaptive/b")
        + _scores(10, "adaptive/a")
        + _scores(10, "adaptive/b")
    )
    c = build(
        [_run(8, "v1", "adaptive"), _run(9, "v1", "adaptive")],
        [_run(10, "v2", "adaptive")],
        scores,
        SIMULATED,
    )
    runs = {r.scenario_id: r.champion.run_ids for r in c.scenarios}
    assert runs == {"adaptive/a": (8,), "adaptive/b": (9,)}
    assert c.not_compared == ["champion: adaptive/b (run #8: infrastructure error)"]


def test_scenario_missing_on_one_side_is_listed_not_compared() -> None:
    scores = _scores(1, "base/a") + _scores(1, "base/b") + _scores(2, "base/a")
    c = build([_run(1, "v1")], [_run(2, "v2")], scores, SIMULATED)
    assert [r.scenario_id for r in c.scenarios] == ["base/a"]
    assert c.not_compared == ["base/b (no result on the challenger side)"]


@pytest.mark.parametrize(
    "field", ["agent_model_id", "judge_model_id", "rubric_version", "dataset_hash", "k"]
)
def test_refuses_when_anything_but_the_agent_differs(field: str) -> None:
    scores = _scores(1, "base/a") + _scores(2, "base/a")
    overrides: dict[str, Any] = {field: 3 if field == "k" else "other"}
    challenger = _run(2, "v2", **overrides)
    with pytest.raises(ComparisonError, match=field):
        build([_run(1, "v1")], [challenger], scores, SIMULATED)


def test_refuses_different_simulator_for_a_simulated_suite() -> None:
    scores = _scores(1, "base/a") + _scores(2, "base/a")
    with pytest.raises(ComparisonError, match="simulator_version"):
        build([_run(1, "v1")], [_run(2, "v2", simulator_version="sim-v1")], scores, SIMULATED)


def test_scripted_suite_ignores_the_simulator_version() -> None:
    # v1 scripted run #5 is a rescore carrying sim-v1; scripted lines never use the simulator.
    scores = _scores(5, "adversarial/x") + _scores(6, "adversarial/x")
    c = build(
        [_run(5, "v1", "adversarial", simulator_version="sim-v1")],
        [_run(6, "v2", "adversarial")],
        scores,
        SIMULATED,
    )
    assert [r.verdict for r in c.scenarios] == ["tie"]


def test_refuses_same_version_or_mixed_versions() -> None:
    scores = _scores(1, "base/a") + _scores(2, "base/a")
    with pytest.raises(ComparisonError, match="both sides"):
        build([_run(1, "v1")], [_run(2, "v1")], scores, SIMULATED)
    with pytest.raises(ComparisonError, match="mix agent versions"):
        build([_run(1, "v1"), _run(3, "v2")], [_run(2, "v2")], scores, SIMULATED)


def test_report_states_combined_effect_label_and_n() -> None:
    scores = _scores(1, "base/a") + _scores(2, "base/a", outcome=False)
    text = render(build([_run(1, "v1")], [_run(2, "v2")], scores, SIMULATED))
    assert "**Label: SYNTHETIC**" in text
    assert "`v2` is measured as one combined change** of 4 fixes" in text
    assert "do not say which fix caused which difference" in text
    assert "| n | 1 scenarios × k=1 per side |" in text
    assert "- `base/a`: outcome 1/1 → 0/1, critical 0 → 0" in text


def test_refuses_aborted_runs() -> None:
    scores = _scores(1, "base/a") + _scores(2, "base/a")
    aborted = _run(2, "v2", status="aborted", status_reason="D-006: v2 identity fix defect")
    with pytest.raises(ComparisonError, match=r"run #2 is aborted \(D-006"):
        build([_run(1, "v1")], [aborted], scores, SIMULATED)


def test_rules_change_is_refused_unless_allowed_and_then_reported() -> None:
    scores = _scores(1, "base/a") + _scores(2, "base/a")
    v3 = _run(2, "v3", rules_version="rules-v7")
    with pytest.raises(ComparisonError, match="rules_version differs"):
        build([_run(1, "v1")], [v3], scores, SIMULATED)
    c = build([_run(1, "v1")], [v3], scores, SIMULATED, ("rules-v6", "rules-v7"))
    assert c.rules_change == ("rules-v6", "rules-v7")
    text = render(c)
    assert "**Rules differ: `rules-v6` → `rules-v7`**" in text
    assert "--allow-rules-change rules-v6:rules-v7" in text


def test_allowed_rules_change_covers_only_that_pair() -> None:
    scores = _scores(1, "base/a") + _scores(2, "base/a")
    v3 = _run(2, "v3", rules_version="rules-v8")
    with pytest.raises(ComparisonError, match="rules_version differs"):
        build([_run(1, "v1")], [v3], scores, SIMULATED, ("rules-v6", "rules-v7"))


def test_unused_rules_change_is_not_reported() -> None:
    scores = _scores(1, "base/a") + _scores(2, "base/a")
    c = build([_run(1, "v1")], [_run(2, "v2")], scores, SIMULATED, ("rules-v6", "rules-v7"))
    assert c.rules_change is None and "Rules differ" not in render(c)


@pytest.mark.parametrize("bad", ["rules-v6", "rules-v6:", ":rules-v7", "rules-v6:rules-v6"])
def test_rules_change_flag_must_be_old_new(bad: str) -> None:
    with pytest.raises(ValueError, match="OLD:NEW"):
        parse_rules_change(bad)
