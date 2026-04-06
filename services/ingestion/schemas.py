"""Response schemas for the ingestion API."""

from typing import Any, Dict

from pydantic import BaseModel, Field


class IngestionResponse(BaseModel):
    success: bool
    message: str
    result_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="file_id, file_name, file_path from process_single_file",
    )
    collection_name: str = Field(
        default="",
        description="Target collection used in the request",
    )
