import datetime
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime

class KakawinEntry(BaseModel):
    """Schema for a Kakawin Ramayana entry."""
    sargah_number: int
    sargah_name: str
    bait: int
    sanskrit_text: str
    text: str


class ChatRequest(BaseModel):
    """Schema for chat request body."""
    query: str
    top_k: int = Field(default=3, gt=0, description="Number of top entries to retrieve")
    context_size: int = Field(default=10, ge=0, description="Number of surrounding entries to include")
    embedding_model: int = Field(default=1, ge=1, le=2, description="Embedding model for RAG (1=Cohere, 2=SentenceTransformers")
    session_id: Optional[str] = None


class ContextEntry(BaseModel):
    """Schema for each context entry in the response."""
    sargah_number: int
    sargah_name: str
    bait: int
    sanskrit_text: str
    text: str
    is_top_k: bool


class ChatResponse(BaseModel):
    """Schema for chat response."""
    response: str
    context: List[ContextEntry] = []
    session_id: str


class ErrorResponse(BaseModel):
    """Schema for error response."""
    error: str