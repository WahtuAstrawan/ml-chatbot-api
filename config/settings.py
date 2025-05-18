import os
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    # Application Settings
    APP_NAME: str = "Kakawin Ramayana Chatbot"
    APP_DESCRIPTION: str = "API for question answering about Kakawin Ramayana"

    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")

    # Dataset
    DATASET_PATH: str = "datasets/dataset_with_embedding.json"

    # Retrieval Settings
    DEFAULT_TOP_K: int = 3
    DEFAULT_CONTEXT_SIZE: int = 10

    # Model Settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    LLM_MODEL_NAME: str = "gemini-2.0-flash"

    # Project root directory
    ROOT_DIR: Path = Path(__file__).parent.parent


@lru_cache()
def get_settings():
    """Cache and return settings to avoid reloading."""
    return Settings()