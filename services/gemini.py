import logging
from functools import lru_cache
from typing import List, Dict, Any

from google import genai

from config.settings import get_settings
from utils.prompt_builder import build_chat_prompt, build_query_enhancement_prompt

logger = logging.getLogger(__name__)
settings = get_settings()


class GeminiService:
    """Service for interacting with Google's Gemini AI."""

    def __init__(self):
        """Initialize the Gemini client."""
        try:
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise

    def enhance_query_with_context(self, query: str, conversation_context: str = "") -> str:
        """
        Enhance the query for better retrieval results with conversation context.

        Args:
            query: Original query string
            conversation_context: Previous conversation context

        Returns:
            Enhanced query string
        """
        try:
            prompt = build_query_enhancement_prompt(query, conversation_context)
            response = self.client.models.generate_content(
                model=settings.LLM_MODEL_NAME,
                contents=prompt
            )
            enhanced = response.text.strip()
            logger.info(f"Gemini enhanced query with context: {enhanced}")
            return enhanced
        except Exception as e:
            logger.warning(f"Failed to enhance query: {str(e)}. Using original query.")
            return query

    def generate_response_with_history(self, query: str, contexts: list,
                                       conversation_history: List[Dict] = None) -> str:
        """
        Generate a response using Gemini AI with conversation history.

        Args:
            query: Original query string
            contexts: List of context entries from RAG
            conversation_history: Previous conversation messages

        Returns:
            Generated response text
        """
        try:
            prompt = build_chat_prompt(query, contexts, conversation_history)
            response = self.client.models.generate_content(
                model=settings.LLM_MODEL_NAME,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise

    # Backward compatibility
    def enhance_query(self, query: str) -> str:
        return self.enhance_query_with_context(query)

    def generate_response(self, query: str, contexts: list) -> str:
        return self.generate_response_with_history(query, contexts)


@lru_cache()
def get_gemini_service() -> GeminiService:
    """Get or create a singleton instance of GeminiService."""
    return GeminiService()