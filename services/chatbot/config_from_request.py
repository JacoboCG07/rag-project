"""Build ChatbotConfig: retrieval from the request body (same as `/search`)."""

from src.chatbot.config import ChatbotConfig

from services.retrieval.config_from_request import build_search_pipeline_config

from .schemas import ChatRequest


def build_chatbot_config(req: ChatRequest) -> ChatbotConfig:
    retrieval = build_search_pipeline_config(req)
    return ChatbotConfig(
        retrieval=retrieval,
        llm_model=req.llm_model,
    )
