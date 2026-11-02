"""Run eval suites against the agent and store every score.

  docker compose exec api python -m evals.run --suites base adversarial adaptive --k 1

One eval_run per suite. Each scenario: run (retry once on infrastructure error), score from
the audit log, judge, record. Then a markdown report is written to reports/.
"""

import argparse
import asyncio
import re
import zlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.agent.versions import load_version, version_path
from app.config import load_settings
from app.db.models import EvalRun, EvalScore, EvalSuite, RunStatus, Speaker, Turn
from app.db.session import make_engine
from app.llm.groq_chat import GroqChat, Usage
from app.logging_setup import configure_pipecat_logging
from app.models_config import ModelsConfig, load_models
from app.policy.config import PolicyRules, load_rules
from evals import checks
from evals.judge import RUBRIC_PATH, JudgeResult, Rubric, as_evidence, judge, load_rubric
from evals.report import write_report
from evals.runner import ConversationResult, run_scenario
from evals.scenario import Scenario, load_suite_scenarios, suite_paths
from evals.store import finish_run, record_score, start_run

INFRA_RETRY_WAIT_S = 60.0  # when the error gives no wait time
MAX_INFRA_RETRIES = 8
WAIT_BUFFER_S = 15.0
_TRY_AGAIN = re.compile(r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.I)


def retry_after_seconds(errors: list[str]) -> float:
    """Groq's own 'Please try again in 2m29.04s' (rate limit), else a default wait."""
    for error in errors:
        m = _TRY_AGAIN.search(error)
        if m and any(m.groups()):
            hours, minutes, seconds = (float(g) if g else 0.0 for g in m.groups())
            return hours * 3600 + minutes * 60 + seconds + WAIT_BUFFER_S
    return INFRA_RETRY_WAIT_S


@dataclass
class Totals:
    agent: Usage = field(default_factory=Usage)
    simulator: Usage = field(default_factory=Usage)
    judge: Usage = field(default_factory=Usage)

    def as_dict(self) -> dict[str, dict[str, int]]:
        return {
            "agent": self.agent.as_dict(),
            "simulator": self.simulator.as_dict(),
            "judge": self.judge.as_dict(),
        }


def scenario_seed(base_seed: int, scenario_id: str, run_index: int) -> int:
    return base_seed + zlib.crc32(scenario_id.encode()) % 100_000 + run_index * 1_000_003


def now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Context:
    engine: AsyncEngine
    rules: PolicyRules
    models: ModelsConfig
    rubric: Rubric
    chat: GroqChat
    agent_api_key: str
    agent_version: str


async def _run_once(ctx: Context, s: Scenario, i: int, seed: int) -> ConversationResult:
    return await run_scenario(
        s,
        i,
        engine=ctx.engine,
        rules=ctx.rules,
        models=ctx.models,
        chat=ctx.chat,
        agent_api_key=ctx.agent_api_key,
        agent_version=ctx.agent_version,
        seed=seed,
    )


async def _delivered_turns(
    engine: AsyncEngine, cid: int
) -> tuple[list[tuple[str, str]], list[str]]:
    """All delivered turns (for context) and the agent lines the LLM wrote (for violations)."""
    async with engine.connect() as conn:
        rows = list(
            await conn.execute(
                select(Turn.speaker, Turn.text, Turn.llm_model)
                .where(Turn.conversation_id == cid)
                .order_by(Turn.seq)
            )
        )
    turns = [("agent" if r.speaker is Speaker.AGENT else "debtor", r.text) for r in rows]
    llm_lines = [r.text for r in rows if r.speaker is Speaker.AGENT and r.llm_model != "script"]
    return turns, llm_lines


async def converse(ctx: Context, s: Scenario, i: int, totals: Totals) -> ConversationResult:
    """Run the scenario; on infrastructure errors wait as long as Groq asks and resume."""
    seed = scenario_seed(42, s.id, i)
    result = await _run_once(ctx, s, i, seed)
    retries = 0
    while result.infra_errors and retries < MAX_INFRA_RETRIES:
        # Groq free-tier limits are the ceiling: wait as long as Groq asks, then resume.
        wait = retry_after_seconds(result.infra_errors)
        retries += 1
        print(
            f"    infra error ({result.infra_errors[0][:70]}); waiting {wait:.0f}s "
            f"(retry {retries}/{MAX_INFRA_RETRIES})",
            flush=True,
        )
        totals.simulator.add_usage(result.debtor_usage)
        await asyncio.sleep(wait)
        result = await _run_once(ctx, s, i, seed)
    totals.simulator.add_usage(result.debtor_usage)
    return result


async def score_and_record(
    ctx: Context,
    run_id: int,
    s: Scenario,
    i: int,
    result: ConversationResult,
    totals: Totals,
    *,
    count_agent_tokens: bool = True,
) -> str:
    """Deterministic checks + judge, stored as eval_scores of this run."""
    seed = scenario_seed(42, s.id, i)
    async with ctx.engine.connect() as conn:
        scored = await checks.score(conn, s, result, ctx.rules, now())
    judged = None
    if result.conversation_id is not None:
        turns, llm_lines = await _delivered_turns(ctx.engine, result.conversation_id)
        if any(who == "debtor" for who, _ in turns):
            judged = await judge(
                ctx.chat, ctx.models.judge.model_id, ctx.rubric, turns, seed, llm_lines
            )
            for u in judged.usage:
                totals.judge.add(u)
            scored.critical += [
                checks.Critical("judge_language_violation", v["type"], {"quote": v["quote"]})
                for v in judged.violations
            ]
        for m in scored.metrics:
            if m.name == "agent_tokens" and count_agent_tokens:
                ev = m.evidence
                totals.agent.calls += int(ev["llm_calls"])
                totals.agent.prompt_tokens += int(ev["prompt_tokens"])
                totals.agent.completion_tokens += int(ev["completion_tokens"])

    async with ctx.engine.begin() as conn:
        for m in scored.metrics:
            await record_score(
                conn,
                run_id=run_id,
                scenario_id=s.id,
                run_index=i,
                metric=m.name,
                value=m.value,
                passed=m.passed,
                conversation_id=result.conversation_id,
                evidence=m.evidence,
            )
        await record_score(
            conn,
            run_id=run_id,
            scenario_id=s.id,
            run_index=i,
            metric="critical_violations",
            value=len(scored.critical),
            passed=not scored.critical,
            conversation_id=result.conversation_id,
            evidence={"items": [c.__dict__ for c in scored.critical]},
        )
        if judged is not None:
            await _record_judged(conn, run_id, s.id, i, result.conversation_id, judged)
    flag = "OK " if not scored.critical else "CRIT"
    return (
        f"{flag} {s.id:<44} outcome={result.outcome} ended_by={result.ended_by} "
        f"critical={len(scored.critical)} near_misses={len(scored.near_misses)}"
    )


async def _record_judged(
    conn: AsyncConnection,
    run_id: int,
    scenario_id: str,
    i: int,
    conversation_id: int | None,
    judged: JudgeResult,
) -> None:
    for name, (score, reason) in judged.scores.items():
        await record_score(
            conn,
            run_id=run_id,
            scenario_id=scenario_id,
            run_index=i,
            metric=f"judge_{name}",
            value=score,
            conversation_id=conversation_id,
            evidence={"reason": reason},
        )
    await record_score(
        conn,
        run_id=run_id,
        scenario_id=scenario_id,
        run_index=i,
        metric="judge_error",
        value=1.0 if judged.error else 0.0,
        passed=judged.error is None,
        conversation_id=conversation_id,
        evidence={"error": judged.error, **as_evidence(judged)},
    )


async def run_suite(ctx: Context, suite: EvalSuite, k: int, only: set[str]) -> int:
    scenarios = [s for s in load_suite_scenarios(suite) if not only or s.id in only]
    async with ctx.engine.begin() as conn:
        run_id = await start_run(
            conn,
            suite=suite,
            agent_version=ctx.agent_version,
            k=k,
            rubric_version=ctx.rubric.version,
            dataset_files=suite_paths(suite),
            models=ctx.models,
            rules=ctx.rules,
            now=now(),
            config_extra=[RUBRIC_PATH, version_path(ctx.agent_version)],
            seed=42,
        )
    print(f"eval_run {run_id}: suite={suite.value} scenarios={len(scenarios)} k={k}")
    totals = Totals()
    status = RunStatus.FAILED
    try:
        for s in scenarios:
            for i in range(k):
                result = await converse(ctx, s, i, totals)
                line = await score_and_record(ctx, run_id, s, i, result, totals)
                print("  " + line, flush=True)
        status = RunStatus.COMPLETED
    finally:
        async with ctx.engine.begin() as conn:
            await finish_run(conn, run_id, status=status, usage=totals.as_dict(), now=now())
    return run_id


async def rescore_run(ctx: Context, source_run_id: int) -> int:
    """Re-score a stored run's conversations with the current checks and rubric.

    No conversation is replayed and no agent tokens are spent. The new run keeps the source
    run's agent and simulator versions (they produced the conversations) and records
    rescored_from_run_id."""
    async with ctx.engine.connect() as conn:
        source = (
            await conn.execute(select(EvalRun.__table__).where(EvalRun.id == source_run_id))
        ).one()
        rows = list(
            await conn.execute(
                select(EvalScore.__table__).where(
                    EvalScore.eval_run_id == source_run_id,
                    EvalScore.metric.in_(["outcome_match", "infra_error"]),
                )
            )
        )
    suite = EvalSuite(source.suite)
    scenarios = {sc.id: sc for sc in load_suite_scenarios(suite)}
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for r in rows:
        by_key.setdefault((r.scenario_id, r.run_index), {})[r.metric] = r
    async with ctx.engine.begin() as conn:
        run_id = await start_run(
            conn,
            suite=suite,
            agent_version=source.agent_version,
            k=source.k,
            rubric_version=ctx.rubric.version,
            dataset_files=suite_paths(suite),
            models=ctx.models,
            rules=ctx.rules,
            now=now(),
            config_extra=[RUBRIC_PATH, version_path(source.agent_version)],
            seed=source.seed,
            simulator_version=source.simulator_version,
            rescored_from_run_id=source_run_id,
        )
    print(f"eval_run {run_id}: rescoring run #{source_run_id} ({suite.value}, {len(by_key)})")
    totals = Totals()
    status = RunStatus.FAILED
    try:
        for (sid, idx), metrics in sorted(by_key.items()):
            om, infra = metrics["outcome_match"], metrics.get("infra_error")
            result = ConversationResult(
                sid,
                idx,
                om.conversation_id,
                om.evidence.get("actual"),
                ended_by=om.evidence.get("ended_by", ""),
                infra_errors=list(infra.evidence.get("errors", [])) if infra else [],
                debtor_raw=list(om.evidence.get("debtor_raw", [])),
                dropped_hangups=list(om.evidence.get("dropped_hangups", [])),
            )
            line = await score_and_record(
                ctx, run_id, scenarios[sid], idx, result, totals, count_agent_tokens=False
            )
            print("  " + line, flush=True)
        status = RunStatus.COMPLETED
    finally:
        async with ctx.engine.begin() as conn:
            await finish_run(conn, run_id, status=status, usage=totals.as_dict(), now=now())
    return run_id


async def rejudge_run(ctx: Context, source_run_id: int) -> int | None:
    """Re-judge only the conversations whose judge failed in a stored run (judge tokens only).

    The new run (rescored_from_run_id) copies every other score of the source run unchanged;
    for a re-judged conversation only judge_* / judge_error / critical_violations are new
    (critical = the source's deterministic items + any judge violation). The source run is
    never changed. None if the source run has no judge error."""
    async with ctx.engine.connect() as conn:
        source = (
            await conn.execute(select(EvalRun.__table__).where(EvalRun.id == source_run_id))
        ).one()
        rows = list(
            await conn.execute(
                select(EvalScore.__table__)
                .where(EvalScore.eval_run_id == source_run_id)
                .order_by(EvalScore.id)
            )
        )
    if source.status != RunStatus.COMPLETED.value or source.rescored_from_run_id is not None:
        raise SystemExit(f"run #{source_run_id} is {source.status}, not an original completed run")
    if (source.rubric_version, source.judge_model_id) != (
        ctx.rubric.version,
        ctx.models.judge.model_id,
    ):
        raise SystemExit(
            f"run #{source_run_id}: judge model or rubric changed since; not comparable"
        )
    by_key: dict[tuple[str, int], list[Any]] = {}
    for r in rows:
        by_key.setdefault((r.scenario_id, r.run_index), []).append(r)
    failed = {
        key
        for key, scores in by_key.items()
        if any(s.metric == "judge_error" and s.passed is False for s in scores)
    }
    if not failed:
        print(f"run #{source_run_id}: no judge errors, nothing to re-judge")
        return None
    suite = EvalSuite(source.suite)
    async with ctx.engine.begin() as conn:
        run_id = await start_run(
            conn,
            suite=suite,
            agent_version=source.agent_version,
            k=source.k,
            rubric_version=ctx.rubric.version,
            dataset_files=suite_paths(suite),
            models=ctx.models,
            rules=ctx.rules,
            now=now(),
            config_extra=[RUBRIC_PATH, version_path(source.agent_version)],
            seed=source.seed,
            simulator_version=source.simulator_version,
            rescored_from_run_id=source_run_id,
        )
    print(f"eval_run {run_id}: re-judging {len(failed)} of run #{source_run_id}")
    totals = Totals()
    status = RunStatus.FAILED
    try:
        for key, scores in sorted(by_key.items()):
            sid, idx = key
            rejudge = key in failed
            judged = None
            if rejudge:
                cid = scores[0].conversation_id
                turns, llm_lines = await _delivered_turns(ctx.engine, cid)
                seed = scenario_seed(42, sid, idx)
                judged = await judge(
                    ctx.chat, ctx.models.judge.model_id, ctx.rubric, turns, seed, llm_lines
                )
                for u in judged.usage:
                    totals.judge.add(u)
            async with ctx.engine.begin() as conn:
                for s in scores:
                    rejudged = s.metric.startswith("judge_") or s.metric == "critical_violations"
                    if rejudge and rejudged:
                        continue
                    await record_score(
                        conn, run_id=run_id, scenario_id=sid, run_index=idx, metric=s.metric,
                        value=s.value, passed=s.passed, conversation_id=s.conversation_id,
                        evidence=s.evidence,
                    )  # fmt: skip
                if judged is not None:
                    source_crit = next(s for s in scores if s.metric == "critical_violations")
                    items = list(source_crit.evidence.get("items", []))
                    items += [
                        checks.Critical(
                            "judge_language_violation", v["type"], {"quote": v["quote"]}
                        ).__dict__
                        for v in judged.violations
                    ]
                    await record_score(
                        conn, run_id=run_id, scenario_id=sid, run_index=idx,
                        metric="critical_violations", value=len(items), passed=not items,
                        conversation_id=source_crit.conversation_id, evidence={"items": items},
                    )  # fmt: skip
                    await _record_judged(
                        conn, run_id, sid, idx, source_crit.conversation_id, judged
                    )
            if judged is not None:
                state = (
                    "recovered" if judged.error is None else f"still failed ({judged.error[:60]})"
                )
                print(f"  {sid:<44} [{idx}] {state}", flush=True)
        status = RunStatus.COMPLETED
    finally:
        async with ctx.engine.begin() as conn:
            await finish_run(conn, run_id, status=status, usage=totals.as_dict(), now=now())
    return run_id


async def main(
    suites: list[EvalSuite],
    k: int,
    agent_version: str,
    report_dir: Path,
    only: set[str],
    rescore: list[int],
    report_runs: list[int],
    rejudge: list[int] = (),  # type: ignore[assignment]
) -> None:
    settings = load_settings()
    rules = load_rules()
    version = load_version(agent_version)
    if version.rules_version != rules.version:
        raise SystemExit(
            f"agent {agent_version} expects {version.rules_version}, rules.yaml is {rules.version}"
        )
    engine = make_engine(settings.app_url)
    models = load_models()
    ctx = Context(
        engine=engine,
        rules=rules,
        models=models,
        rubric=load_rubric(),
        chat=GroqChat(settings.groq_api_key),
        agent_api_key=settings.agent_api_key(models.agent.provider),
        agent_version=agent_version,
    )
    run_ids = [await rescore_run(ctx, source) for source in rescore]
    rejudged = [await rejudge_run(ctx, source) for source in rejudge]
    run_ids += [r for r in rejudged if r is not None]
    run_ids += [await run_suite(ctx, suite, k, only) for suite in suites]
    run_ids = sorted(set(run_ids) | set(report_runs))
    path = await write_report(engine, run_ids, report_dir)
    print(f"report: {path}")
    await engine.dispose()


if __name__ == "__main__":
    configure_pipecat_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suites", nargs="*", default=None, choices=[s.value for s in EvalSuite],
        help="default: base adversarial adaptive, unless only re-scoring or re-judging",
    )  # fmt: skip
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--agent-version", default="v1")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--only", nargs="*", default=[], help="scenario ids to run (default: all)")
    parser.add_argument(
        "--rescore-run", type=int, nargs="*", default=[],
        help="re-score these stored runs with the current checks and rubric (no replay)",
    )  # fmt: skip
    parser.add_argument(
        "--report-runs", type=int, nargs="*", default=[],
        help="also include these existing runs in the report",
    )  # fmt: skip
    parser.add_argument(
        "--rejudge-errors", type=int, nargs="*", default=[],
        help="re-judge only the judge_error conversations of these stored runs (judge tokens only)",
    )  # fmt: skip
    args = parser.parse_args()
    if args.suites is None:
        only_stored = args.rescore_run or args.rejudge_errors
        args.suites = [] if only_stored else ["base", "adversarial", "adaptive"]
    asyncio.run(
        main(
            [EvalSuite(s) for s in args.suites],
            args.k,
            args.agent_version,
            args.report_dir,
            set(args.only),
            args.rescore_run,
            args.report_runs,
            args.rejudge_errors,
        )
    )
