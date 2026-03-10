"""
RAG Chatbot - Retrieval + LLM generation pipeline
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

from src.retrieval.pipeline import SearchPipeline
from src.llms.text import OpenAITextModel
from src.utils.utils import PromptLoader
from src.utils import get_logger

from .config import ChatbotConfig


def _format_context(results: List[Dict[str, Any]]) -> str:
    """Format search results as context string for the LLM."""
    if not results:
        return "No se encontró ningún fragmento relevante en la base de documentos."

    parts = []
    for i, r in enumerate(results, 1):
        text = r.get("text", "").strip()
        file_name = r.get("file_name", "N/A")
        pages = r.get("pages", "N/A")
        score = r.get("score", 0)
        if text:
            parts.append(
                f"### Fragmento {i} (Documento: {file_name}, páginas: {pages}, relevancia: {score:.2f})\n{text}"
            )
    return "\n\n".join(parts) if parts else "No hay texto disponible en los resultados."


def _get_rag_prompt(user_query: str, context: str) -> tuple[str, str]:
    """
    Build RAG prompt and system prompt.

    Returns:
        tuple[str, str]: (prompt, system_prompt)
    """
    prompt_path = Path(__file__).parent / "chatbot_prompt.md"
    system_prompt = PromptLoader.read_file(str(prompt_path))
    prompt = f"## Contexto\n\n{context}\n\n## Pregunta del usuario\n\n{user_query}"
    return prompt, system_prompt


class RAGChatbot:
    """
    RAG Chatbot: retrieval + LLM generation.
    Uses SearchPipeline for retrieval and OpenAITextModel for answer generation.
    """

    def __init__(self, config: ChatbotConfig):
        """
        Initialize RAG Chatbot.

        Args:
            config: ChatbotConfig with retrieval and LLM settings.
        """
        self.config = config
        self.logger = get_logger(__name__)
        self._pipeline: Optional[SearchPipeline] = None
        self._llm: Optional[OpenAITextModel] = None

        self._pipeline = SearchPipeline(config=config.retrieval)
        self._llm = OpenAITextModel(
            model=config.llm_model,
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
        )
        self.logger.info(
            "RAGChatbot initialized",
            extra={
                "collection_name": config.retrieval.collection_name,
                "llm_model": config.llm_model,
            },
        )

    @property
    def pipeline(self) -> SearchPipeline:
        """Search pipeline for retrieval."""
        if self._pipeline is None:
            raise RuntimeError("Chatbot has been closed")
        return self._pipeline

    @property
    def llm(self) -> OpenAITextModel:
        """LLM for answer generation."""
        if self._llm is None:
            raise RuntimeError("Chatbot has been closed")
        return self._llm

    def ask(self, query: str) -> str:
        """
        Answer a question using retrieval + LLM generation.

        Args:
            query: User question.

        Returns:
            str: Generated answer.
        """
        self.logger.info("Processing query", extra={"query_length": len(query)})
        results = self.pipeline.search(query=query)
        context = _format_context(results)
        prompt, system_prompt = _get_rag_prompt(user_query=query, context=context)

        system = self.config.system_prompt if self.config.system_prompt else system_prompt
        response = self.llm.call_text_model(
            prompt=prompt,
            system_prompt=system,
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
        )
        return response.strip()

    def close(self) -> None:
        """Close pipeline connections."""
        self.logger.info("Closing RAGChatbot")
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None
        self._llm = None

    def __enter__(self) -> "RAGChatbot":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
