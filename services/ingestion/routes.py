"""HTTP routes for the ingestion service (multipart)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, status

from src.ingestion.config import IngestionPipelineConfig
from src.ingestion.extractors.factory import DocumentExtractorFactory
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.utils import get_logger

from .multipart_parse import normalize_job_id, parse_extract_images
from .schemas import IngestionRequest, IngestionResponse

logger = get_logger(__name__)

router = APIRouter(tags=["ingestion"])

_SUPPORTED = frozenset(DocumentExtractorFactory.get_supported_extensions())


def _safe_filename(upload_name: str) -> str:
    name = Path(upload_name).name
    if not name or name in (".", ".."):
        return "upload.bin"
    return name


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/ingest", response_model=IngestionResponse)
async def ingest(
    body: IngestionRequest = Form(..., media_type="multipart/form-data"),
) -> IngestionResponse:
    try:
        extract_flag = parse_extract_images(body.extract_images)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e

    job_id = normalize_job_id(body.job_id)

    orig_name = body.file.filename or "upload"
    safe_name = _safe_filename(orig_name)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in _SUPPORTED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported type: {suffix or '(no extension)'}. Allowed: {sorted(_SUPPORTED)}",
        )

    try:
        config = IngestionPipelineConfig(collection_name=body.collection_name)
    except ValueError as e:
        logger.warning("Invalid ingestion configuration: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e

    contents = await body.file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The file is empty.",
        )

    tmp_dir: str | None = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="rag_ingest_")
        dest = Path(tmp_dir) / safe_name
        dest.write_bytes(contents)

        with IngestionPipeline(config=config) as pipeline:
            success, message, result_info = pipeline.process_single_file(
                file_path=str(dest.resolve()),
                extract_process_images=extract_flag,
                job_id=job_id,
            )

        return IngestionResponse(
            success=success,
            message=message,
            result_info=result_info or {},
            collection_name=body.collection_name,
        )
    except ValueError as e:
        logger.warning("Ingestion validation error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception("Ingest failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The ingestion service is temporarily unavailable.",
        ) from e
    finally:
        if tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
