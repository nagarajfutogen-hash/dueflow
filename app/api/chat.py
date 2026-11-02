"""Text chat over WebSocket for a conversation created by POST /calls.

Protocol (JSON): client sends {"type":"auth","token":...} first (token never in the URL, so
it stays out of access logs), then {"type":"user","text":...}. Server sends
{"type":"agent","source":"script"|"llm","text":...}, {"type":"tool",...},
{"type":"error",...} and finally {"type":"end","outcome":...}. The client may send
{"type":"hangup"}; a hang-up or dropped connection before any outcome is `disconnected`.
"""

import asyncio
import contextlib
import secrets
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agent.live import LiveChat

router = APIRouter()

AUTH_TIMEOUT_S = 10.0
POLICY_VIOLATION = 1008


async def _authenticate(ws: WebSocket) -> bool:
    expected: str = ws.app.state.settings.api_token
    try:
        first = await asyncio.wait_for(ws.receive_json(), AUTH_TIMEOUT_S)
    except (TimeoutError, ValueError, WebSocketDisconnect):
        return False
    token = first.get("token") if isinstance(first, dict) else None
    return (
        bool(expected)
        and isinstance(token, str)
        and first.get("type") == "auth"
        and secrets.compare_digest(token.encode(), expected.encode())
    )


@router.websocket("/chat/ws")
async def chat_ws(ws: WebSocket, conversation_id: int) -> None:
    await ws.accept()
    if not await _authenticate(ws):
        await ws.close(code=POLICY_VIOLATION, reason="unauthorized")
        return

    async def send(message: dict[str, Any]) -> None:
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await ws.send_json(message)

    state = ws.app.state
    live = await LiveChat.start(
        conversation_id=conversation_id,
        engine=state.engine,
        rules=state.rules,
        models=state.models,
        api_key=state.settings.agent_api_key(state.models.agent.provider),
        send=send,
    )
    if live is None:
        await ws.close(code=POLICY_VIOLATION, reason="conversation not available")
        return

    ended = asyncio.create_task(live.session.ended.wait())
    try:
        while True:
            receive = asyncio.create_task(ws.receive_json())
            done, _ = await asyncio.wait({receive, ended}, return_when=asyncio.FIRST_COMPLETED)
            if ended in done:
                receive.cancel()
                break
            message = receive.result()
            if isinstance(message, dict) and message.get("type") == "hangup":
                break
            if not isinstance(message, dict) or message.get("type") != "user":
                await send({"type": "error", "error": 'expected {"type":"user","text":"..."}'})
                continue
            await live.debtor_says(str(message.get("text", "")))
    except (WebSocketDisconnect, ValueError, RuntimeError):
        pass
    finally:
        ended.cancel()
        # Shielded: teardown must finish even if this handler is cancelled (server stop).
        await asyncio.shield(asyncio.ensure_future(_close(ws, live)))


async def _close(ws: WebSocket, live: LiveChat) -> None:
    await live.close()
    # The client may already have hung up; closing again is not an error.
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        await ws.close()
