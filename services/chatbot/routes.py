"""HTTP routes for the chatbot service."""

from fastapi import APIRouter, HTTPException, status

from src.chatbot.config_from_request import build_chatbot_config
from src.chatbot.rag_chatbot import RAGChatbot
from src.utils import get_logger
from src.utils.document_store import document_url

from .schemas import ChatRequest, ChatResponse, ChatSource

logger = get_logger(__name__)

router = APIRouter(tags=["chatbot"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    try:
        config = build_chatbot_config(
            search_type=body.search_type,
            collection_name=body.collection_name,
            search_limit=body.search_limit,
            llm_model=body.llm_model,
        )
    except ValueError as e:
        logger.warning("Invalid chatbot configuration: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e

    try:
        with RAGChatbot(config=config) as bot:
            answer, cited = bot.ask_with_sources(query=body.query, job_id=body.job_id)
        sources = [
            ChatSource(
                file_name=c["file_name"],
                pages=c["pages"],
                url=document_url(
                    body.collection_name, c["file_name"], c["pages"][0] if c["pages"] else None
                ),
            )
            for c in cited
        ]
        return ChatResponse(answer=answer, sources=sources)
    except ValueError as e:
        logger.warning("Chat validation error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception("Chat failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The chat service is temporarily unavailable.",
        ) from e
