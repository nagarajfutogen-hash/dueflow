# CLAUDE.md — dueflow

Collections conversation agent. Full design: `PLANNING.md` (source of truth). Read it before any task.
Ask me when this file and `PLANNING.md` conflict. Do not change `PLANNING.md` without asking.

## Workflow
- Local first: build and validate everything locally in Docker. No production setup until I say so.
- One milestone at a time. Plan first, wait for my approval, then implement.
- I handle all git commits and pushes. Do not run git commands.
- A task is done only when tests pass, `ruff` and `mypy` are clean, and the task's acceptance criteria are met.
- End each task with: what was built, how to run it, what is left open.

## Keep it lean
- Simplest working solution first. No extra services, abstractions or features beyond the current task.
- Dropped: Langfuse, Twilio, Smart Turn, Gemini/NVIDIA fallbacks, Lab version diff, audio playback.
- If something feels complex, stop and ask before building it.
- Pre-approved dependencies: `PLANNING.md` §2 plus pydantic, httpx, sqlalchemy, alembic, asyncpg, pyyaml, pytest, pytest-asyncio, ruff, mypy. Ask for anything else.

## Hard rules
1. **The LLM proposes, the policy engine decides.** `app/policy/` is pure Python: no LLM, no network, no DB writes inside rule functions.
2. Money terms, disclosure, history access and contact rules happen only through tools validated by the policy engine.
3. Every policy decision (allowed or blocked) writes an `events` row with rule, input, decision, reason.
4. `events` is append-only. Never update or delete it.
5. Never hardcode, mock or invent metrics. Every displayed number comes from a stored eval run.
6. Label results SYNTHETIC or HUMAN TEST. Never present them as production outcomes.
7. Secrets only in `.env`; keep `.env.example` updated.
8. Postgres is never exposed publicly.
9. Debtor data is synthetic but treated as real PII: redact it in logs.
10. Model IDs live in `config/models.yaml` (committed), never in code or `.env`; `.env` holds secrets only.
    Groq is used only when needed. If a Groq rate limit is reached (429 / tokens per day) or LLM
    performance degrades (timeouts, latency, error rate), tell me right away: I will provide a
    new key. Never print or log keys. OpenAI is used only for final k=3 testing, with a key I provide. Use Groq for development; OpenAI only for final eval runs with my approval.
11. Clock is injectable; never call the system time directly in policy code.

## Frontend (Next.js)
- `output: 'export'` static build. No API routes, no server actions, no middleware auth. All logic lives in FastAPI.

## Code standards
- Type hints everywhere; async for I/O; `uv` for Python.
- Tests for every module; policy changes need allowed AND blocked cases.
- Timeouts and retries on every external call.
- JPY amounts as integers; timezones from each debtor's own field.

## Commands
- Run: `docker compose up -d`
- Tests: `docker compose exec api pytest`
- Lint/types: `docker compose exec api ruff check . && docker compose exec api mypy app evals tests`
- Migrations: `docker compose exec api alembic upgrade head`
