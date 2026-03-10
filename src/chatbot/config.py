"""
Configuration for RAG Chatbot
"""

from typing import Optional
from pydantic import BaseModel, Field

from src.retrieval.config import SearchPipelineConfig


class ChatbotConfig(BaseModel):
    """Configuration for RAG Chatbot (retrieval + LLM generation)."""

    retrieval: SearchPipelineConfig = Field(
        ...,
        description="Search pipeline config: collection_name, search_type, search_limit, etc."
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for answer generation (e.g. gpt-4o-mini, gpt-4o)"
    )
    llm_max_tokens: int = Field(
        default=1000,
        ge=1,
        le=16_384,
        description="Maximum tokens for LLM response"
    )
    llm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for LLM generation"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Override system prompt for RAG (if None, loads from chatbot_prompt.md)"
    )
