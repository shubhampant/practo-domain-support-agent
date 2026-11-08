
"""Simple in-memory conversation storage with session isolation."""

from __future__ import annotations

from threading import RLock
from typing import Any


class SessionMemory:
    """Store conversation turns separately for each session."""

    def __init__(self, max_turns: int = 20) -> None:
        if not isinstance(max_turns, int) or isinstance(max_turns, bool):
            raise TypeError("max_turns must be an integer.")
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")

        self.max_turns = max_turns
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._lock = RLock()

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Append a complete user/assistant exchange."""
        session_id = self._validate_text(session_id, "session_id")
        user_message = self._validate_text(user_message, "user_message")
        assistant_message = self._validate_text(
            assistant_message, "assistant_message"
        )

        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.extend(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ]
            )

            # Keep the most recent complete exchanges.
            max_messages = self.max_turns * 2
            if len(history) > max_messages:
                del history[:-max_messages]

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Return a copy of the session's conversation history."""
        session_id = self._validate_text(session_id, "session_id")

        with self._lock:
            return [
                message.copy()
                for message in self._sessions.get(session_id, [])
            ]

    def format_history(self, session_id: str) -> str:
        """Format previous turns for use as conversational context."""
        history = self.get_history(session_id)

        if not history:
            return "No previous conversation."

        return "\n".join(
            f"{message['role'].capitalize()}: {message['content']}"
            for message in history
        )

    def clear_session(self, session_id: str) -> bool:
        """Delete one session's history; return whether it existed."""
        session_id = self._validate_text(session_id, "session_id")

        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def clear_all(self) -> None:
        """Delete all sessions from this in-memory store."""
        with self._lock:
            self._sessions.clear()

    @staticmethod
    def _validate_text(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")
        return value.strip()
