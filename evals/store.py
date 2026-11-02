"""Eval run lifecycle: start a run (with every version and fingerprint), add scores, finish."""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import insert, update
from sqlalchemy.ext.asyncio import AsyncConnection

from app.agent.scripts import SCRIPTS_VERSION
from app.agent.versions import load_version
from app.db.models import EvalRun, EvalScore, EvalSuite, ResultLabel, RunStatus
from app.models_config import ModelsConfig
from app.policy.config import PolicyRules
from evals.fingerprint import REPO_ROOT, config_hash, files_hash, git_sha
from evals.simulate import SIMULATOR_VERSION


async def start_run(
    conn: AsyncConnection,
    *,
    suite: EvalSuite,
    agent_version: str,
    k: int,
    rubric_version: str,
    dataset_files: Sequence[Path],
    models: ModelsConfig,
    rules: PolicyRules,
    now: datetime,
    config_extra: Sequence[Path] = (),
    seed: int | None = None,
    root: Path = REPO_ROOT,
    simulator_version: str = SIMULATOR_VERSION,
    rescored_from_run_id: int | None = None,
) -> int:
    """Create a running eval run. Every number later shown traces back to this row."""
    result = await conn.execute(
        insert(EvalRun)
        .values(
            suite=suite,
            label=ResultLabel.SYNTHETIC,
            agent_version=agent_version,
            agent_model_id=models.agent.model_id,
            simulator_model_id=models.simulator.model_id,
            simulator_version=simulator_version,
            rescored_from_run_id=rescored_from_run_id,
            judge_model_id=models.judge.model_id,
            rubric_version=rubric_version,
            rules_version=rules.version,
            prompt_version=load_version(agent_version).prompt_version,
            scripts_version=SCRIPTS_VERSION,
            config_hash=config_hash(config_extra, root),
            dataset_hash=files_hash(dataset_files, root),
            dataset_git_sha=git_sha(root),
            k=k,
            seed=seed,
            status=RunStatus.RUNNING,
            started_at=now,
        )
        .returning(EvalRun.id)
    )
    return result.scalar_one()


async def record_score(
    conn: AsyncConnection,
    *,
    run_id: int,
    scenario_id: str,
    run_index: int,
    metric: str,
    value: float,
    passed: bool | None = None,
    conversation_id: int | None = None,
    evidence: dict[str, Any] | None = None,
) -> int:
    result = await conn.execute(
        insert(EvalScore)
        .values(
            eval_run_id=run_id,
            scenario_id=scenario_id,
            run_index=run_index,
            conversation_id=conversation_id,
            metric=metric,
            value=value,
            passed=passed,
            evidence=evidence or {},
        )
        .returning(EvalScore.id)
    )
    return result.scalar_one()


async def finish_run(
    conn: AsyncConnection,
    run_id: int,
    *,
    status: RunStatus,
    usage: dict[str, Any],
    now: datetime,
) -> None:
    await conn.execute(
        update(EvalRun)
        .where(EvalRun.id == run_id)
        .values(status=status, usage=usage, finished_at=now)
    )
