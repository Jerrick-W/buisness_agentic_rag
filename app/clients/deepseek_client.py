"""DeepSeek API client module.

Wraps DeepSeek chat completion (streaming / non-streaming) and embedding
generation behind a thin async client with exponential-backoff retry.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

_BASE_DELAY = 1.0  # seconds
_MAX_RETRIES = 3


def _should_retry(exc: Exception) -> bool:
    """Return True when the error is transient and worth retrying."""
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.ConnectError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429:
            return True  # rate-limited – retry
        if 400 <= code < 500:
            return False  # other 4xx – do NOT retry
        if code >= 500:
            return True  # 5xx – retry
    return True  # network-level / unknown errors – retry


def _build_error(error_type: str, description: str) -> dict[str, str]:
    """Build a structured error dict."""
    return {"error_type": error_type, "description": description}


# ---------------------------------------------------------------------------
# DeepSeekClient
# ---------------------------------------------------------------------------

class DeepSeekClient:
    """Async client for the DeepSeek API (OpenAI-compatible)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self._settings = settings or Settings()  # type: ignore[call-arg]
        self._api_key = api_key or self._settings.deepseek_api_key
        self._base_url = (base_url or self._settings.deepseek_base_url).rstrip("/")
        self._max_retries = max_retries
        self._chat_model = self._settings.deepseek_chat_model

        # Embedding uses Qwen via DashScope (separate API key & base URL)
        self._embedding_api_key = self._settings.embedding_api_key or self._api_key
        self._embedding_base_url = self._settings.embedding_base_url.rstrip("/")
        self._embedding_model = self._settings.embedding_model

    # -- internal helpers ---------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any],
        *,
        stream: bool = False,
        timeout: float = 60.0,
    ) -> httpx.Response | dict[str, str]:
        """Send an HTTP request with exponential-backoff retry.

        Returns the *httpx.Response* on success or a structured error dict
        when all retries are exhausted.
        """
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                if stream:
                    # For streaming we need a client that stays open.
                    client = httpx.AsyncClient(timeout=timeout)
                    req = client.build_request(method, url, headers=self._headers(), json=json_body)
                    resp = await client.send(req, stream=True)
                    resp.raise_for_status()
                    # Attach client so caller can close it later.
                    resp._client = client  # type: ignore[attr-defined]
                    return resp
                else:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        resp = await client.request(method, url, headers=self._headers(), json=json_body)
                        resp.raise_for_status()
                        return resp
            except Exception as exc:
                last_exc = exc
                if not _should_retry(exc):
                    logger.warning("Non-retryable error on attempt %d: %s", attempt + 1, exc)
                    break
                if attempt < self._max_retries - 1:
                    delay = _BASE_DELAY * (2 ** attempt)  # 1s, 2s, 4s
                    logger.warning(
                        "Retryable error on attempt %d, retrying in %.1fs: %s",
                        attempt + 1,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted (or non-retryable error).
        return _build_error(
            error_type=type(last_exc).__name__ if last_exc else "UnknownError",
            description=str(last_exc) if last_exc else "Request failed after retries",
        )

    # -- public API ---------------------------------------------------------

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str | AsyncGenerator[str, None] | dict[str, str]:
        """Call the DeepSeek chat completion endpoint.

        Returns:
            - *str* – full assistant reply (non-streaming).
            - *AsyncGenerator[str, None]* – yields tokens (streaming).
            - *dict* with ``error_type`` / ``description`` on failure.
        """
        body: dict[str, Any] = {
            "model": self._chat_model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        result = await self._request_with_retry("POST", "/v1/chat/completions", body, stream=stream)

        if isinstance(result, dict):
            # Structured error from retry logic.
            return result

        resp: httpx.Response = result

        if stream:
            return self._iter_stream(resp)
        else:
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def _iter_stream(self, resp: httpx.Response) -> AsyncGenerator[str, None]:
        """Yield tokens from an SSE stream response."""
        import json as _json

        client: httpx.AsyncClient | None = getattr(resp, "_client", None)
        try:
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = _json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except _json.JSONDecodeError:
                    continue
        finally:
            await resp.aclose()
            if client is not None:
                await client.aclose()

    async def create_embedding(self, text: str) -> list[float] | dict[str, str]:
        """Generate an embedding vector via Qwen (DashScope OpenAI-compatible API).

        Returns a list of floats on success or a structured error dict on
        failure.
        """
        body = {
            "model": self._embedding_model,
            "input": text,
        }
        url = f"{self._embedding_base_url}/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self._embedding_api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(url, headers=headers, json=body)
                    resp.raise_for_status()
                    data = resp.json()
                    return data["data"][0]["embedding"]
            except Exception as exc:
                last_exc = exc
                if not _should_retry(exc):
                    logger.warning("Embedding non-retryable error on attempt %d: %s", attempt + 1, exc)
                    break
                if attempt < self._max_retries - 1:
                    delay = _BASE_DELAY * (2 ** attempt)
                    logger.warning("Embedding retryable error on attempt %d, retrying in %.1fs: %s", attempt + 1, delay, exc)
                    await asyncio.sleep(delay)

        return _build_error(
            error_type=type(last_exc).__name__ if last_exc else "UnknownError",
            description=str(last_exc) if last_exc else "Embedding request failed after retries",
        )
