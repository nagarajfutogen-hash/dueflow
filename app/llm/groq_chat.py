"""Minimal Groq chat-completions client for evals (simulator, attackers, judge).

Rate limits are Groq's own free-tier limits: on 429 / 5xx / timeouts it waits (Retry-After
when given, else exponential backoff) and resumes, instead of failing the eval run.
"""

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLMUnavailableError(RuntimeError):
    pass


class InvalidJSONError(LLMUnavailableError):
    """JSON mode: Groq could not produce valid JSON (HTTP 400 json_validate_failed). Unlike
    other 4xx errors, a re-sample with another seed can succeed."""


@dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, result: ChatResult) -> None:
        self.calls += 1
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens

    def add_usage(self, other: "Usage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class GroqChat:
    api_key: str
    client: httpx.AsyncClient = field(
        default_factory=lambda: httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    )
    max_attempts: int = 8
    max_wait_s: float = 90.0
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    url: str = GROQ_URL

    def _wait(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return min(self.max_wait_s, float(retry_after) + 0.5)
                except ValueError:
                    pass
        backoff: float = 2.0 * 2.0**attempt
        return min(self.max_wait_s, backoff)

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        seed: int | None = None,
        max_tokens: int = 400,
        json_mode: bool = False,
        reasoning_effort: str | None = "low",
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error = "no attempt made"
        for attempt in range(self.max_attempts):
            response: httpx.Response | None = None
            try:
                response = await self.client.post(self.url, json=body, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = type(exc).__name__
            else:
                if response.status_code == 200:
                    data = response.json()
                    usage = data.get("usage") or {}
                    text = data["choices"][0]["message"].get("content") or ""
                    return ChatResult(
                        _THINK.sub("", text).strip(),
                        int(usage.get("prompt_tokens", 0)),
                        int(usage.get("completion_tokens", 0)),
                    )
                last_error = f"HTTP {response.status_code}"
                if response.status_code == 400 and "json_validate_failed" in response.text:
                    raise InvalidJSONError(f"{last_error}: {response.text[:200]}")
                if response.status_code != 429 and response.status_code < 500:
                    raise LLMUnavailableError(f"{last_error}: {response.text[:200]}")
            await self.sleep(self._wait(attempt, response))
        raise LLMUnavailableError(f"gave up after {self.max_attempts} attempts ({last_error})")
