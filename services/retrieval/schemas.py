"""Pydantic schemas for the retrieval API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.retrieval.config import SearchType


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query text")
    collection_name: str = Field(..., min_length=1, description="Milvus collection")
    search_type: SearchType = Field(
        default=SearchType.SIMPLE,
        description="Search strategy (simple, with_selection, with_selection_and_metadata)",
    )
    search_limit: int = Field(
        default=10,
        ge=5,
        le=25,
        description="Maximum number of results",
    )
    job_id: Optional[str] = Field(
        default=None,
        description="Job id for log tracing",
    )


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(
        ...,
        description="Matching chunks (text, score, file_name, pages, etc.)",
    )
