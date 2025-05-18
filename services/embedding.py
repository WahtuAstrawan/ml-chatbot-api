import logging
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingService:
    """Service for embedding text using sentence transformers."""

    def __init__(self):
        """Initialize the embedding model."""
        try:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of texts to encode

        Returns:
            Array of embeddings
        """
        try:
            return self.model.encode(texts)
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Get or create a singleton instance of EmbeddingService."""
    return EmbeddingService()