.PHONY: up down reset-dev migrate seed test lint eval

up:
	docker compose up -d

down:
	docker compose down

# Wipes the dev database (events included), then migrates and seeds. Local only.
reset-dev:
	docker compose down -v
	docker compose up -d --wait
	docker compose exec -T api alembic upgrade head
	docker compose exec -T api python -m app.db.seed

migrate:
	docker compose exec -T api alembic upgrade head

seed:
	docker compose exec -T api python -m app.db.seed

# Offline only. Live LLM tests need RUN_LIVE=1 (Groq; report token usage). OpenAI: ask first.
test:
	docker compose exec -T api pytest -q

lint:
	docker compose exec -T api ruff check .
	docker compose exec -T api ruff format --check .
	docker compose exec -T api mypy app evals tests

# Full eval pass (Groq; free-tier rate limits apply: the runner waits and resumes).
# SUITES / K / AGENT_VERSION can be overridden: make eval SUITES="base" K=3
SUITES ?= base adversarial adaptive
K ?= 1
AGENT_VERSION ?= v1
eval:
	docker compose exec -T api python -m evals.run --suites $(SUITES) --k $(K) --agent-version $(AGENT_VERSION)
