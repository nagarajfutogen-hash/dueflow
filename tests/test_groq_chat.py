import json

import httpx
import pytest

from app.llm.groq_chat import GroqChat, InvalidJSONError, LLMUnavailableError, Usage

OK_BODY = {
    "choices": [{"message": {"content": "<think>hmm</think> Hello there."}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
}


def _client(
    responses: list[httpx.Response], seen: list[dict[str, object]], waits: list[float] | None = None
) -> GroqChat:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return queue.pop(0)

    recorded = waits if waits is not None else []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    return GroqChat(
        "key", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), sleep=fake_sleep
    )


async def test_success_strips_think_and_counts_tokens() -> None:
    seen: list[dict[str, object]] = []
    chat = _client([httpx.Response(200, json=OK_BODY)], seen)
    result = await chat.complete(model="m", messages=[{"role": "user", "content": "hi"}], seed=7)
    assert result.text == "Hello there."
    assert (result.prompt_tokens, result.completion_tokens) == (12, 3)
    assert seen[0]["seed"] == 7 and seen[0]["model"] == "m"
    usage = Usage()
    usage.add(result)
    assert usage.as_dict() == {"calls": 1, "prompt_tokens": 12, "completion_tokens": 3}


async def test_rate_limit_waits_retry_after_then_resumes() -> None:
    waits: list[float] = []
    chat = _client(
        [
            httpx.Response(429, headers={"retry-after": "7"}),
            httpx.Response(503),
            httpx.Response(200, json=OK_BODY),
        ],
        [],
        waits,
    )
    result = await chat.complete(model="m", messages=[])
    assert result.text == "Hello there."
    assert waits == [7.5, 4.0]


async def test_client_error_fails_fast() -> None:
    chat = _client([httpx.Response(400, text="bad request")], [])
    with pytest.raises(LLMUnavailableError, match="HTTP 400"):
        await chat.complete(model="m", messages=[])


async def test_groq_invalid_json_is_its_own_error() -> None:
    body = '{"error":{"message":"Failed to generate JSON.","code":"json_validate_failed"}}'
    seen: list[dict[str, object]] = []
    chat = _client([httpx.Response(400, text=body)], seen)
    with pytest.raises(InvalidJSONError):
        await chat.complete(model="m", messages=[], json_mode=True)
    assert len(seen) == 1  # no blind resend: the caller re-samples with a new seed


async def test_other_400s_are_not_invalid_json() -> None:
    chat = _client([httpx.Response(400, text="bad request")], [])
    with pytest.raises(LLMUnavailableError) as exc:
        await chat.complete(model="m", messages=[])
    assert not isinstance(exc.value, InvalidJSONError)


async def test_gives_up_after_max_attempts() -> None:
    chat = _client([httpx.Response(429) for _ in range(3)], [])
    chat.max_attempts = 3
    with pytest.raises(LLMUnavailableError, match="gave up after 3"):
        await chat.complete(model="m", messages=[])


async def test_json_mode_sets_response_format() -> None:
    seen: list[dict[str, object]] = []
    chat = _client([httpx.Response(200, json=OK_BODY)], seen)
    await chat.complete(model="m", messages=[], json_mode=True, reasoning_effort=None)
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in seen[0]
