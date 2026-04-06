"""Pydantic schemas for the chatbot API."""

from pydantic import BaseModel, Field

from services.retrieval.schemas import SearchRequest


class ChatRequest(SearchRequest):
    """Extends the `POST /search` body (`SearchRequest`) with the chat model."""

    llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model used to generate the answer",
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Answer generated from RAG context")
