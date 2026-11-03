"""Build ChatbotConfig from chat API parameters."""

from src.chatbot.config import ChatbotConfig
from src.retrieval.config import SearchType
from src.retrieval.config_builder import build_search_pipeline_config


def build_chatbot_config(
    *,
    search_type: SearchType,
    collection_name: str,
    search_limit: int,
    llm_model: str | None,
) -> ChatbotConfig:
    retrieval = build_search_pipeline_config(
        search_type=search_type,
        collection_name=collection_name,
        search_limit=search_limit,
    )
    return ChatbotConfig(retrieval=retrieval, llm_model=llm_model)
