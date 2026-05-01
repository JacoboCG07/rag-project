"""Build SearchPipelineConfig from API-style parameters (HTTP services)."""

from src.retrieval.config import SearchPipelineConfig, SearchType


def build_search_pipeline_config(
    *,
    search_type: SearchType,
    collection_name: str,
    search_limit: int,
) -> SearchPipelineConfig:
    return SearchPipelineConfig(
        search_type=search_type,
        collection_name=collection_name,
        search_limit=search_limit,
    )
