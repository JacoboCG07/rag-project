"""FastAPI entry point for the ingestion service."""

from fastapi import FastAPI

from .routes import router

app = FastAPI(
    title="RAG Ingestion",
    version="1.0.0",
    description=(
        "Multipart upload of documents (PDF/TXT) into Milvus. "
        "Runtime options are defined by IngestionPipelineConfig. "
        "Form fields: file, collection_name, extract_images, job_id."
    ),
)

app.include_router(router)
