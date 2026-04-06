"""Build IngestionPipelineConfig with per-request collection_name and INGEST_* settings."""

import os

from src.ingestion.config import IngestionPipelineConfig
from src.llms.embeddings.openai_embedder import OpenAIEmbedder


def build_ingestion_pipeline_config(collection_name: str) -> IngestionPipelineConfig:
    """
    Operator parameters via INGEST_*; each request sets collection_name.
    Optional variables: INGEST_CHUNK_SIZE, INGEST_CHUNK_OVERLAP, EMBEDDING_MODEL,
    INGEST_MAX_IMAGES_PER_DOCUMENT.
    """
    kwargs: dict = {"collection_name": collection_name}

    raw_chunk = os.getenv("INGEST_CHUNK_SIZE", "").strip()
    if raw_chunk:
        kwargs["chunk_size"] = int(raw_chunk)

    raw_overlap = os.getenv("INGEST_CHUNK_OVERLAP", "").strip()
    if raw_overlap:
        kwargs["chunk_overlap"] = int(raw_overlap)

    raw_emb = os.getenv("EMBEDDING_MODEL", "").strip()
    if raw_emb:
        kwargs["embedder"] = OpenAIEmbedder(model=raw_emb, count_tokens=False)

    raw_max = os.getenv("INGEST_MAX_IMAGES_PER_DOCUMENT", "").strip()
    if raw_max:
        kwargs["max_images_per_document"] = int(raw_max)

    return IngestionPipelineConfig(**kwargs)
