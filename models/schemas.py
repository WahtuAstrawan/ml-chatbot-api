from pydantic import BaseModel, Field
from typing import List


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


class ErrorResponse(BaseModel):
    """Schema for error response."""
    error: str