"""Request and response schemas for the ingestion API."""

from typing import Annotated, Any, Dict

from pydantic import BaseModel, Field
from fastapi import File, Form, UploadFile


class IngestionRequest(BaseModel):
    """Multipart body for `POST /ingest` (file + form fields).

    The route must declare this parameter as
    ``Form(..., media_type="multipart/form-data")`` so FastAPI parses multipart
    instead of expecting JSON.
    """

    file: Annotated[UploadFile, File(..., description="PDF or TXT")]
    collection_name: Annotated[str, Form(...), Field(min_length=1)]
    extract_images: Annotated[str, Form(..., description="true or false")]
    job_id: Annotated[str | None, Form(default=None)] = None


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
