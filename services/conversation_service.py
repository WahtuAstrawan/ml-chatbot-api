import uuid
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict
from models.schemas import ChatMessage
import logging

logger = logging.getLogger(__name__)


class ConversationService:
    """Service for managing conversation history."""

    def __init__(self, max_history: int = 10):
        """
        Initialize conversation service.

        Args:
            max_history: Maximum number of messages to keep in history
        """
        self.conversations: Dict[str, List[ChatMessage]] = defaultdict(list)
        self.max_history = max_history

    def create_session(self) -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        self.conversations[session_id] = []
        logger.info(f"Created new session: {session_id}")
        return session_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Add a message to conversation history.

        Args:
            session_id: Session identifier
            role: "user" or "assistant"
            content: Message content
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now()
        )

        self.conversations[session_id].append(message)

        # Keep only recent messages
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]

        logger.debug(f"Added {role} message to session {session_id}")

    def get_conversation_history(self, session_id: str) -> List[ChatMessage]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of chat messages
        """
        return self.conversations.get(session_id, [])

    def get_context_for_query(self, session_id: str, current_query: str) -> str:
        """
        Build context string from conversation history for better query understanding.

        Args:
            session_id: Session identifier
            current_query: Current user query

        Returns:
            Context string for query enhancement
        """
        history = self.get_conversation_history(session_id)
        if not history:
            return current_query

        # Build context from recent messages (last 6 messages)
        recent_messages = history[-6:] if len(history) > 6 else history
        context_parts = []

        for msg in recent_messages:
            if msg.role == "user":
                context_parts.append(f"User sebelumnya bertanya: {msg.content}")
            else:
                context_parts.append(f"Assistant menjawab: {msg.content}")

        context_parts.append(f"Pertanyaan sekarang: {current_query}")

        return "\n".join(context_parts)


# Singleton instance
_conversation_service = None

def get_conversation_service() -> ConversationService:
    """Get or create singleton instance of ConversationService."""
    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationService()
    return _conversation_service