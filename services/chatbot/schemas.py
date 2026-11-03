"""Pydantic schemas for the chatbot API."""

from pydantic import BaseModel, Field

from services.retrieval.schemas import SearchRequest


class ChatRequest(SearchRequest):
    """Extends the `POST /search` body (`SearchRequest`) with the chat model."""

    llm_model: str | None = Field(
        default=None,
        description="Model used to generate the answer (on Azure: deployment name). Empty = provider default",
    )


class ChatSource(BaseModel):
    file_name: str = Field(..., description="Cited document")
    pages: list[str] = Field(default_factory=list, description="Cited pages of the document")
    url: str = Field(..., description="Relative URL of the original file, opened at the first cited page")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Answer generated from RAG context")
    sources: list[ChatSource] = Field(
        default_factory=list, description="Documents and pages cited by the answer"
    )
