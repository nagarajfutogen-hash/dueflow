import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.staticfiles import StaticFiles

from app.api.admin import login_router
from app.api.admin import router as admin_router
from app.api.calls import router as calls_router
from app.api.chat import router as chat_router
from app.api.debtors import router as debtors_router
from app.api.lab import router as lab_router
from app.api.voice import router as voice_router
from app.config import load_settings
from app.db.session import make_engine, ping
from app.logging_setup import configure_pipecat_logging
from app.models_config import load_models
from app.policy.config import load_rules
from app.voice.spike import router as voice_spike_router

configure_pipecat_logging()

DEV_PAGE_DIR = Path(__file__).resolve().parent.parent / "web" / "dev"
ADMIN_UI_DIR = Path(__file__).resolve().parent.parent / "web" / "out"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    app.state.settings = settings
    app.state.engine = make_engine(settings.app_url)
    app.state.rules = load_rules()
    app.state.models = load_models()
    app.state.voice_tasks = set()  # running voice calls (and the Day 5 spike)
    app.state.replay_task = None  # the Failure Lab replay running in this process, if any
    yield
    for task in [*app.state.voice_tasks, app.state.replay_task]:
        if task is not None:
            task.cancel()
    await app.state.engine.dispose()


app = FastAPI(title="dueflow", lifespan=lifespan)
app.include_router(calls_router)
app.include_router(chat_router)
app.include_router(debtors_router)
app.include_router(voice_router)
# Dev-only Day 5 WebRTC spike: echo, no LLM. Removed once the VPS spike has passed.
app.include_router(voice_spike_router)
app.include_router(lab_router)
app.include_router(login_router)
app.include_router(admin_router)
# Dev-only bare chat page for manual testing. The real UI is the Next.js app (Day 5-6).
app.mount("/dev", StaticFiles(directory=DEV_PAGE_DIR, html=True), name="dev")
# Admin dashboard + Failure Lab UI: the Next.js static export (web/out), same origin as the
# API. Built with `docker compose run --rm web`. Not built (or not in the image): a clear 404
# and one startup warning, never a 500 (VPS, 2026-11-03).
if ADMIN_UI_DIR.is_dir():
    app.mount("/admin-ui", StaticFiles(directory=ADMIN_UI_DIR, html=True), name="admin-ui")
else:
    logging.getLogger(__name__).warning("admin UI not built: %s is missing", ADMIN_UI_DIR)

    @app.get("/admin-ui/{path:path}", include_in_schema=False)
    async def admin_ui_missing(path: str) -> None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "admin UI not built (run: docker compose run --rm web)"
        )


@app.get("/health")
async def health(request: Request, response: Response) -> dict[str, str]:
    db_ok = await ping(request.app.state.engine)
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if db_ok else "degraded", "db": "ok" if db_ok else "unreachable"}
