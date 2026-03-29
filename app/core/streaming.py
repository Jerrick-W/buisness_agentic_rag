"""Streaming engine module.

Provides SSE (Server-Sent Events) streaming for chat responses.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

from app.clients.deepseek_client import DeepSeekClient
from app.models import DocumentSource

logger = logging.getLogger(__name__)


def _sse_payload(event_type: str, content: str | list | dict) -> dict:
    """Build an SSE event dict for EventSourceResponse."""
    return {"data": json.dumps({"type": event_type, "content": content}, ensure_ascii=False)}


class StreamingEngine:
    """SSE-based streaming output engine."""

    def __init__(self, deepseek_client: DeepSeekClient) -> None:
        self._deepseek = deepseek_client

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        sources: list[DocumentSource] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion tokens.

        Yields JSON strings (EventSourceResponse handles SSE framing):
        - type=token: individual token content
        - type=sources: document sources used (sent before done)
        - type=done: stream complete marker
        - type=error: error description
        """
        try:
            result = await self._deepseek.chat_completion(messages, stream=True)

            # Check for error dict
            if isinstance(result, dict):
                yield _sse_payload("error", result.get("description", "Unknown error"))
                yield _sse_payload("done", "[DONE]")
                return

            # Stream tokens
            async for token in result:
                yield _sse_payload("token", token)

            # Send sources if available
            if sources:
                sources_data = [s.model_dump() for s in sources]
                yield _sse_payload("sources", sources_data)

            # Done marker
            yield _sse_payload("done", "[DONE]")

        except Exception as exc:
            logger.error("Streaming error: %s", exc)
            yield _sse_payload("error", str(exc))
            yield _sse_payload("done", "[DONE]")

    async def non_stream_chat(
        self,
        messages: list[dict[str, str]],
    ) -> str | dict[str, str]:
        """Non-streaming chat completion. Returns full response text or error dict."""
        return await self._deepseek.chat_completion(messages, stream=False)
