"""Pydantic schemas for the chatbot API."""

from pydantic import BaseModel, Field

from services.retrieval.schemas import SearchRequest


class ChatRequest(SearchRequest):
    """Extends the `POST /search` body (`SearchRequest`) with the chat model."""

    llm_model: str | None = Field(
        default=None,
        description="Model used to generate the answer (on Azure: deployment name). Empty = provider default",
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Answer generated from RAG context")
