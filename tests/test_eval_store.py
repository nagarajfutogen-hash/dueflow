from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.models import EvalRun, EvalSuite, RunStatus
from app.models_config import load_models
from app.policy.config import load_rules
from evals.fingerprint import REPO_ROOT, config_hash, files_hash, git_sha
from evals.simulate import SIMULATOR_VERSION
from evals.store import finish_run, record_score, start_run

NOW = datetime(2026, 10, 24, 6, tzinfo=UTC)
MODELS = load_models()
RULES = load_rules()
DATASET = [REPO_ROOT / "config" / "models.yaml"]


def _tree(tmp_path: Path, **files: str) -> Path:
    for rel, content in files.items():
        path = tmp_path / rel.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


# --- Fingerprints ---------------------------------------------------------------------------


def test_files_hash_is_order_independent_and_content_sensitive(tmp_path: Path) -> None:
    root = _tree(tmp_path, a__x="1", b__y="2")
    a, b = root / "a/x", root / "b/y"
    assert files_hash([a, b], root) == files_hash([b, a], root)
    before = files_hash([a, b], root)
    a.write_text("changed", encoding="utf-8")
    assert files_hash([a, b], root) != before


def test_files_hash_includes_the_path(tmp_path: Path) -> None:
    root = _tree(tmp_path, one="same", two="same")
    assert files_hash([root / "one"], root) != files_hash([root / "two"], root)


def test_config_hash_changes_when_rules_change(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        config__models_yaml="m",
        app__policy__rules_yaml="r1",
        app__agent__nodes_py="n",
        app__agent__scripts_py="s",
    )
    for rel in ("config/models_yaml", "app/policy/rules_yaml"):
        (root / rel).rename(root / rel.replace("_yaml", ".yaml"))
    for rel in ("app/agent/nodes_py", "app/agent/scripts_py"):
        (root / rel).rename(root / rel.replace("_py", ".py"))
    before = config_hash(root=root)
    (root / "app/policy/rules.yaml").write_text("r2", encoding="utf-8")
    assert config_hash(root=root) != before


def test_git_sha_none_without_repository(tmp_path: Path) -> None:
    assert git_sha(tmp_path) is None


def test_git_sha_from_loose_ref(tmp_path: Path) -> None:
    _tree(tmp_path, **{".git__HEAD": "ref: refs/heads/main\n", ".git__refs__heads__main": "a" * 40})
    assert git_sha(tmp_path) == "a" * 40


def test_git_sha_from_packed_refs(tmp_path: Path) -> None:
    _tree(
        tmp_path,
        **{
            ".git__HEAD": "ref: refs/heads/main\n",
            ".git__packed-refs": f"# pack-refs\n{'b' * 40} refs/heads/main\n",
        },
    )
    assert git_sha(tmp_path) == "b" * 40


def test_git_sha_unborn_branch_is_none(tmp_path: Path) -> None:
    _tree(tmp_path, **{".git__HEAD": "ref: refs/heads/main\n"})
    assert git_sha(tmp_path) is None


# --- Run lifecycle (app role, test DB) ------------------------------------------------------


async def _start(engine: AsyncEngine, dataset: list[Path]) -> int:
    async with engine.begin() as conn:
        return await start_run(
            conn,
            suite=EvalSuite.BASE,
            agent_version="v1",
            k=1,
            rubric_version="rubric-test",
            dataset_files=dataset,
            models=MODELS,
            rules=RULES,
            now=NOW,
        )


async def test_run_records_models_versions_and_fingerprints(app_engine: AsyncEngine) -> None:
    run_id = await _start(app_engine, DATASET)
    async with app_engine.connect() as conn:
        run = (await conn.execute(select(EvalRun.__table__).where(EvalRun.id == run_id))).one()
    assert run.agent_model_id == MODELS.agent.model_id
    assert run.simulator_model_id == MODELS.simulator.model_id
    assert run.simulator_version == SIMULATOR_VERSION
    assert run.judge_model_id == MODELS.judge.model_id
    assert run.rules_version == RULES.version
    assert run.label == "SYNTHETIC"
    assert run.status == "running"
    assert len(run.config_hash) == 64 and len(run.dataset_hash) == 64
    assert run.dataset_git_sha is None  # project is not a git repository yet


async def test_scores_and_finish(app_engine: AsyncEngine) -> None:
    run_id = await _start(app_engine, DATASET)
    async with app_engine.begin() as conn:
        await record_score(
            conn,
            run_id=run_id,
            scenario_id="base/cooperative",
            run_index=0,
            metric="task_completion",
            value=1.0,
            passed=True,
            evidence={"event_ids": [1, 2]},
        )
        await finish_run(
            conn,
            run_id,
            status=RunStatus.COMPLETED,
            usage={"agent": {"prompt_tokens": 10, "completion_tokens": 2}},
            now=NOW,
        )
    async with app_engine.connect() as conn:
        run = (await conn.execute(select(EvalRun.__table__).where(EvalRun.id == run_id))).one()
    assert run.status == "completed" and run.finished_at == NOW
    assert run.usage["agent"]["prompt_tokens"] == 10


async def test_duplicate_score_rejected(app_engine: AsyncEngine) -> None:
    run_id = await _start(app_engine, DATASET)
    async with app_engine.connect() as conn:
        await record_score(conn, run_id=run_id, scenario_id="s", run_index=0, metric="m", value=1)
        with pytest.raises(IntegrityError):
            await record_score(
                conn, run_id=run_id, scenario_id="s", run_index=0, metric="m", value=0
            )
        await conn.rollback()


async def test_app_cannot_delete_eval_rows(app_engine: AsyncEngine) -> None:
    async with app_engine.connect() as conn:
        with pytest.raises(DBAPIError, match="permission denied"):
            await conn.execute(text("DELETE FROM eval_scores"))
        await conn.rollback()
