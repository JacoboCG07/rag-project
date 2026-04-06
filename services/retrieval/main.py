"""FastAPI entry point for the retrieval service."""

from fastapi import FastAPI

from .routes import router

app = FastAPI(
    title="RAG Retrieval",
    version="1.0.0",
    description=(
        "Vector search over Milvus. "
        "The embedding model must match the collection dimension."
    ),
)

app.include_router(router)
