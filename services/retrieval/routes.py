"""HTTP routes for the retrieval service."""

from fastapi import APIRouter, HTTPException, status

from src.retrieval.config_builder import build_search_pipeline_config
from src.retrieval.pipeline import SearchPipeline
from src.utils import get_logger

from .schemas import SearchRequest, SearchResponse

logger = get_logger(__name__)

router = APIRouter(tags=["retrieval"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest) -> SearchResponse:
    try:
        config = build_search_pipeline_config(
            search_type=body.search_type,
            collection_name=body.collection_name,
            search_limit=body.search_limit,
        )
    except ValueError as e:
        logger.warning("Invalid search configuration: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e

    try:
        with SearchPipeline(config=config) as pipeline:
            results = pipeline.search(
                query=body.query,
                filter_expr=None,
                job_id=body.job_id,
            )
        return SearchResponse(results=results)
    except ValueError as e:
        logger.warning("Search validation error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception("Search failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The search service is temporarily unavailable.",
        ) from e
