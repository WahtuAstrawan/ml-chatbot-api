import logging
from functools import lru_cache
from typing import List
import numpy as np
import cohere

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CohereEmbeddingService:
    def __init__(self):
        try:
            self.client = cohere.Client(settings.COHERE_API_KEY)
            self.model_name = settings.COHERE_EMBEDDING_MODEL_NAME
            logger.info(f"Cohere model '{self.model_name}' initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {str(e)}")
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        try:
            response = self.client.embed(texts=texts, model=self.model_name, input_type="search_document")
            return np.array(response.embeddings)
        except Exception as e:
            logger.error(f"Error getting embeddings from Cohere: {str(e)}")
            raise


@lru_cache()
def get_cohere_embedding_service() -> CohereEmbeddingService:
    """Get or create a singleton instance of CohereEmbeddingService."""
    return CohereEmbeddingService()