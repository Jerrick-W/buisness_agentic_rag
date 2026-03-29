"""Unit tests for the DeepSeek client module."""

from __future__ import annotations

import json
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest
import pytest_asyncio

from app.clients.deepseek_client import DeepSeekClient, _should_retry, _build_error


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _env_api_key(monkeypatch):
    """Ensure DEEPSEEK_API_KEY is set for Settings validation."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-123")


@pytest.fixture
def client(_env_api_key):
    """Return a DeepSeekClient wired to a fake base URL."""
    return DeepSeekClient(api_key="test-key-123", base_url="https://fake.api")


# ---------------------------------------------------------------------------
# _should_retry
# ---------------------------------------------------------------------------

class TestShouldRetry:
    def test_timeout_is_retryable(self):
        assert _should_retry(httpx.ReadTimeout("timeout")) is True

    def test_connect_error_is_retryable(self):
        assert _should_retry(httpx.ConnectError("conn refused")) is True

    def test_429_is_retryable(self):
        resp = httpx.Response(429, request=httpx.Request("POST", "https://x"))
        exc = httpx.HTTPStatusError("rate limit", request=resp.request, response=resp)
        assert _should_retry(exc) is True

    def test_500_is_retryable(self):
        resp = httpx.Response(500, request=httpx.Request("POST", "https://x"))
        exc = httpx.HTTPStatusError("server error", request=resp.request, response=resp)
        assert _should_retry(exc) is True

    def test_401_not_retryable(self):
        resp = httpx.Response(401, request=httpx.Request("POST", "https://x"))
        exc = httpx.HTTPStatusError("unauthorized", request=resp.request, response=resp)
        assert _should_retry(exc) is False

    def test_400_not_retryable(self):
        resp = httpx.Response(400, request=httpx.Request("POST", "https://x"))
        exc = httpx.HTTPStatusError("bad request", request=resp.request, response=resp)
        assert _should_retry(exc) is False


# ---------------------------------------------------------------------------
# _build_error
# ---------------------------------------------------------------------------

class TestBuildError:
    def test_structure(self):
        err = _build_error("TimeoutError", "request timed out")
        assert err == {"error_type": "TimeoutError", "description": "request timed out"}


# ---------------------------------------------------------------------------
# Retry behaviour (mocked HTTP)
# ---------------------------------------------------------------------------

class TestRetryBehaviour:
    @pytest.mark.asyncio
    async def test_retries_on_timeout_then_succeeds(self, client):
        """First call times out, second succeeds."""
        ok_resp = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi"}}]},
            request=httpx.Request("POST", "https://fake.api/v1/chat/completions"),
        )

        call_count = 0

        async def _mock_request(method, url, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ReadTimeout("timeout")
            return ok_resp

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.request = _mock_request
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await client.chat_completion([{"role": "user", "content": "hello"}])

        assert result == "hi"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_returns_error_after_max_retries(self, client):
        """All 3 attempts fail → structured error returned."""
        async def _always_timeout(method, url, **kw):
            raise httpx.ReadTimeout("timeout")

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.request = _always_timeout
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await client.chat_completion([{"role": "user", "content": "hello"}])

        assert isinstance(result, dict)
        assert result["error_type"] == "ReadTimeout"
        assert "timeout" in result["description"]
        # Should have slept twice (between attempt 1→2 and 2→3)
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_401(self, client):
        """401 should NOT trigger retry."""
        call_count = 0

        async def _mock_request(method, url, **kw):
            nonlocal call_count
            call_count += 1
            resp = httpx.Response(401, request=httpx.Request("POST", url))
            raise httpx.HTTPStatusError("unauthorized", request=resp.request, response=resp)

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.request = _mock_request
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await client.chat_completion([{"role": "user", "content": "hello"}])

        assert isinstance(result, dict)
        assert result["error_type"] == "HTTPStatusError"
        assert call_count == 1  # no retry

    @pytest.mark.asyncio
    async def test_retry_delays_are_exponential(self, client):
        """Verify sleep durations: 1s then 2s."""
        async def _always_fail(method, url, **kw):
            raise httpx.ConnectError("refused")

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.request = _always_fail
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                await client.chat_completion([{"role": "user", "content": "hello"}])

        delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0]


# ---------------------------------------------------------------------------
# chat_completion (non-streaming)
# ---------------------------------------------------------------------------

class TestChatCompletion:
    @pytest.mark.asyncio
    async def test_non_stream_returns_content(self, client):
        ok_resp = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Hello world"}}]},
            request=httpx.Request("POST", "https://fake.api/v1/chat/completions"),
        )

        async def _mock_request(method, url, **kw):
            return ok_resp

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.request = _mock_request
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await client.chat_completion(
                [{"role": "user", "content": "hi"}], stream=False
            )

        assert result == "Hello world"


# ---------------------------------------------------------------------------
# create_embedding
# ---------------------------------------------------------------------------

class TestCreateEmbedding:
    @pytest.mark.asyncio
    async def test_returns_vector(self, client):
        vector = [0.1] * 1536
        ok_resp = httpx.Response(
            200,
            json={"data": [{"embedding": vector}]},
            request=httpx.Request("POST", "https://fake.api/v1/embeddings"),
        )

        async def _mock_request(method, url, **kw):
            return ok_resp

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.request = _mock_request
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await client.create_embedding("hello")

        assert isinstance(result, list)
        assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_returns_error_on_failure(self, client):
        async def _always_fail(method, url, **kw):
            raise httpx.ReadTimeout("timeout")

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.request = _always_fail
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await client.create_embedding("hello")

        assert isinstance(result, dict)
        assert "error_type" in result
        assert "description" in result
