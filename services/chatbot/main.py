"""FastAPI entry point for the RAG chatbot service."""

from fastapi import FastAPI

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
