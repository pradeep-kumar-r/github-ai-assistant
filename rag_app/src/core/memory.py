"""Simple conversation memory manager for chat sessions."""

from collections import defaultdict
from typing import Any


class ConversationMemory:
    """In-memory conversation storage with session management."""

    def __init__(self, max_history: int = 10):
        """Initialize conversation memory.

        Args:
            max_history: Maximum number of messages to keep per session
        """
        self.max_history = max_history
        self.conversations: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add_message(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> None:
        """Add a message to the conversation history.

        Args:
            session_id: Unique session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional metadata (e.g., sources, tokens)
        """
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }

        self.conversations[session_id].append(message)

        # Keep only the last max_history messages
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """Get conversation history for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            List of messages in the conversation
        """
        return self.conversations.get(session_id, [])

    def get_messages_for_llm(self, session_id: str) -> list[dict[str, str]]:
        """Get formatted messages for LLM API.

        Args:
            session_id: Unique session identifier

        Returns:
            List of messages formatted for LLM (role, content)
        """
        history = self.get_history(session_id)
        return [{"role": msg["role"], "content": msg["content"]} for msg in history]

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session.

        Args:
            session_id: Unique session identifier
        """
        if session_id in self.conversations:
            del self.conversations[session_id]

    def get_active_sessions(self) -> list[str]:
        """Get list of active session IDs."""
        return list(self.conversations.keys())


# Global instance
conversation_memory = ConversationMemory(max_history=20)