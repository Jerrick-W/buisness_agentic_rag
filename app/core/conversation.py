"""Conversation manager module.

Manages multi-turn conversation sessions with in-memory storage.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.models import Message, MessageRole, Session


class ConversationManager:
    """In-memory conversation session manager."""

    def __init__(self, max_context_turns: int = 20) -> None:
        self._sessions: dict[str, Session] = {}
        self._max_context_turns = max_context_turns

    async def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = Session(
            session_id=session_id,
            created_at=datetime.now(timezone.utc),
        )
        return session_id

    async def get_history(self, session_id: str) -> list[Message]:
        """Return full message history for a session."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found")
        return list(session.messages)

    async def add_message(self, session_id: str, message: Message) -> None:
        """Append a message to the session history."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found")
        session.messages.append(message)

    async def get_context_window(self, session_id: str) -> list[Message]:
        """Return the most recent messages within the context window."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found")
        messages = session.messages
        if len(messages) > self._max_context_turns * 2:
            return list(messages[-(self._max_context_turns * 2):])
        return list(messages)

    async def add_user_and_assistant(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        sources: list | None = None,
    ) -> None:
        """Convenience: add a user message and the assistant reply."""
        now = datetime.now(timezone.utc)
        await self.add_message(session_id, Message(
            role=MessageRole.USER,
            content=user_content,
            timestamp=now,
        ))
        await self.add_message(session_id, Message(
            role=MessageRole.ASSISTANT,
            content=assistant_content,
            timestamp=now,
            sources=sources,
        ))

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())
