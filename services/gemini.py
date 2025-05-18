import logging
from functools import lru_cache

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

    def enhance_query(self, query: str) -> str:
        """
        Enhance the query for better retrieval results.

        Args:
            query: Original query string

        Returns:
            Enhanced query string
        """
        try:
            prompt = build_query_enhancement_prompt(query)
            response = self.client.models.generate_content(
                model=settings.LLM_MODEL_NAME,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Failed to enhance query: {str(e)}. Using original query.")
            return query

    def generate_response(self, query: str, contexts: list) -> str:
        """
        Generate a response using Gemini AI.

        Args:
            query: Original query string
            contexts: List of context entries

        Returns:
            Generated response text
        """
        try:
            prompt = build_chat_prompt(query, contexts)
            response = self.client.models.generate_content(
                model=settings.LLM_MODEL_NAME,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise


@lru_cache()
def get_gemini_service() -> GeminiService:
    """Get or create a singleton instance of GeminiService."""
    return GeminiService()