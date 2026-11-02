FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    UV_HTTP_TIMEOUT=300

WORKDIR /srv

COPY pyproject.toml uv.lock ./
# Cache downloads across builds so a flaky network retry doesn't start over.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
