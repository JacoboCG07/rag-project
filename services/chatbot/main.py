"""FastAPI entry point for the RAG chatbot service."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.utils.document_store import documents_dir

from .routes import router

app = FastAPI(
    title="RAG Chatbot",
    version="1.0.0",
    description=(
        "Answers generated with context retrieved from Milvus. "
        "Requires OPENAI_API_KEY (or AZURE_OPENAI_* with LLM_PROVIDER=azure); embedding dimension must match the collection."
    ),
)

app.include_router(router)

# Original documents (written by the ingestion service), so cited sources can be opened.
_docs = documents_dir()
_docs.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=_docs), name="files")
