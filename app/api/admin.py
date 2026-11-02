"""Admin dashboard API (PLANNING §16): login with one admin password, then read-only views."""

import secrets
import time
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app import admin, admin_customers, admin_quality
from app.api.auth import require_token
from app.clock import Clock, SystemClock
from app.db.models import CallbackStatus

LOGIN_TRIES_PER_MINUTE = 5

login_router = APIRouter()
router = APIRouter(prefix="/admin", dependencies=[Depends(require_token)])


def _now(request: Request) -> datetime:
    """Real time for 'today' and 'live'; tests set app.state.admin_clock."""
    clock: Clock = getattr(request.app.state, "admin_clock", None) or SystemClock()
    return clock.now()


class LoginIn(BaseModel):
    password: str = Field(min_length=1, max_length=200)


class LoginOut(BaseModel):
    token: str


def _too_many_tries(request: Request) -> bool:
    """Max LOGIN_TRIES_PER_MINUTE tries per client per minute (kept in memory)."""
    tries: dict[str, list[float]] | None = getattr(request.app.state, "login_tries", None)
    if tries is None:
        tries = request.app.state.login_tries = {}
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    recent = [t for t in tries.get(client, []) if now - t < 60]
    recent.append(now)
    tries[client] = recent
    return len(recent) > LOGIN_TRIES_PER_MINUTE


@login_router.post("/auth/login")
async def login(body: LoginIn, request: Request) -> LoginOut:
    if _too_many_tries(request):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "too many tries, wait a minute")
    settings = request.app.state.settings
    expected: str = settings.admin_password
    if not expected or not settings.api_token:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "admin login is not set up")
    if not secrets.compare_digest(body.password.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "wrong password")
    return LoginOut(token=settings.api_token)


@router.get("/overview")
async def overview(
    request: Request, period: admin.Period = "7d", source: admin.Source = "manual"
) -> dict[str, Any]:
    async with request.app.state.engine.connect() as conn:
        return await admin.overview(conn, period=period, source=source, now=_now(request))


@router.get("/calls")
async def calls(
    request: Request, status: admin.Status = "all", source: admin.Source = "manual"
) -> list[dict[str, Any]]:
    async with request.app.state.engine.connect() as conn:
        return await admin.calls(conn, status=status, source=source, now=_now(request))


@router.get("/calls/{conversation_id}")
async def call_detail(conversation_id: int, request: Request) -> dict[str, Any]:
    async with request.app.state.engine.connect() as conn:
        detail = await admin.call_detail(conn, conversation_id, _now(request))
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "call not found")
    return detail


@router.get("/callbacks")
async def callbacks(request: Request, status: CallbackStatus | None = None) -> list[dict[str, Any]]:
    async with request.app.state.engine.connect() as conn:
        return await admin.callbacks(conn, status=status, now=_now(request))


async def _decide(callback_id: int, request: Request, *, approve: bool) -> dict[str, Any]:
    try:
        async with request.app.state.engine.begin() as conn:
            done = await admin.decide_callback(
                conn, callback_id, approve=approve, now=_now(request)
            )
    except admin.CallbackDecisionError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    if done is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "callback not found")
    return done


@router.post("/callbacks/{callback_id}/approve")
async def approve_callback(callback_id: int, request: Request) -> dict[str, Any]:
    """Only now may auto dial reach the customer (when due; the pre-call check still decides)."""
    return await _decide(callback_id, request, approve=True)


@router.post("/callbacks/{callback_id}/reject")
async def reject_callback(callback_id: int, request: Request) -> dict[str, Any]:
    return await _decide(callback_id, request, approve=False)


# --- D-2: customers, payment calendar, needs attention ---------------------------------------


@router.get("/customers")
async def customers(request: Request) -> list[dict[str, Any]]:
    async with request.app.state.engine.connect() as conn:
        return await admin_customers.customers(conn, now=_now(request))


@router.get("/customers/{ref}")
async def customer(ref: str, request: Request) -> dict[str, Any]:
    async with request.app.state.engine.connect() as conn:
        found = await admin_customers.customer(conn, ref, now=_now(request))
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "customer not found")
    return found


@router.get("/calendar")
async def calendar(
    request: Request, start: date | None = None, days: int = 31
) -> list[dict[str, Any]]:
    now = _now(request)
    first = start or now.astimezone(admin.BUSINESS_TZ).date().replace(day=1)
    async with request.app.state.engine.connect() as conn:
        return await admin_customers.calendar(conn, start=first, days=days, now=now)


@router.get("/attention")
async def attention(request: Request) -> dict[str, Any]:
    async with request.app.state.engine.connect() as conn:
        return await admin_customers.attention(conn, now=_now(request))


# --- D-3: quality (owner view) ---------------------------------------------------------------


@router.get("/quality")
async def quality(
    request: Request,
    period: admin.Period = "all",
    source: admin.Source = "all",
    version: str | None = None,
) -> dict[str, Any]:
    async with request.app.state.engine.connect() as conn:
        return await admin_quality.quality(
            conn, period=period, source=source, version=version or None, now=_now(request)
        )
