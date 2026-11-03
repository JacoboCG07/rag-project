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
    llm_model: Optional[str] = Field(
        default=None,
        description="LLM model for answer generation (on Azure: deployment name). None = provider default"
    )
    llm_max_tokens: int = Field(
        default=4000,
        ge=1,
        le=16_384,
        description="Maximum tokens for LLM response (on reasoning models it also includes reasoning tokens)"
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
