"""
RAG Chatbot - Retrieval + LLM generation pipeline
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.milvus.metadata_tags import parse_tagged
from src.retrieval.pipeline import SearchPipeline
from src.llms import BaseTextModel, create_text_model
from src.utils.utils import PromptLoader
from src.utils import get_logger

from .config import ChatbotConfig


def _format_context(results: List[Dict[str, Any]]) -> str:
    """Format search results as context string for the LLM."""
    if not results:
        return "No se encontró ningún fragmento relevante en la base de documentos."

    # No numbering: the fragment is an internal unit the user never sees, so it must not be cited.
    parts = []
    for r in results:
        text = r.get("text", "").strip()
        if not text:
            continue
        header = [f"Documento: {r.get('file_name') or 'N/A'}"]
        pages = ", ".join(parse_tagged(r.get("pages")))
        if pages:
            header.append(f"páginas: {pages}")
        chapters = ", ".join(parse_tagged(r.get("chapters")))
        if chapters:
            header.append(f"capítulos: {chapters}")
        header.append(f"relevancia: {r.get('score', 0):.2f}")
        parts.append(f"### {' | '.join(header)}\n{text}")
    return "\n\n".join(parts) if parts else "No hay texto disponible en los resultados."


_SOURCES_RE = re.compile(r"<sources>(.*?)</sources>", re.DOTALL | re.IGNORECASE)


def split_sources(
    raw_answer: str, results: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Separate the `<sources>[...]</sources>` block the LLM appends from the visible answer.

    Cited sources are checked against the retrieved results: documents or pages that were not
    in the context are dropped, so a source never points to something the LLM made up.

    Returns:
        (answer without the block, [{"file_name": str, "pages": [str, ...]}, ...]).
        Sources are [] when the block is missing or malformed.
    """
    match = _SOURCES_RE.search(raw_answer)
    answer = _SOURCES_RE.sub("", raw_answer).strip()
    if not match:
        return answer, []
    try:
        cited = json.loads(match.group(1))
    except json.JSONDecodeError:
        return answer, []
    if not isinstance(cited, list):
        return answer, []

    # What was actually retrieved: file_name -> set of pages
    retrieved: Dict[str, set] = {}
    for r in results:
        name = r.get("file_name")
        if name:
            retrieved.setdefault(name, set()).update(parse_tagged(r.get("pages")))

    sources: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}
    for item in cited:
        if not isinstance(item, dict):
            continue
        name = item.get("file_name")
        if name not in retrieved:
            continue
        pages = item.get("pages") or []
        if not isinstance(pages, list):
            pages = [pages]
        valid = [str(p).strip() for p in pages if str(p).strip() in retrieved[name]]
        if name in seen:
            seen[name]["pages"] += [p for p in valid if p not in seen[name]["pages"]]
            continue
        entry = {"file_name": name, "pages": list(dict.fromkeys(valid))}
        seen[name] = entry
        sources.append(entry)

    for entry in sources:
        entry["pages"].sort(key=lambda p: (not p.isdigit(), int(p) if p.isdigit() else 0, p))
    return answer, sources


def _get_rag_prompt(user_query: str, context: str) -> tuple[str, str]:
    """
    Build RAG prompt and system prompt.

    Returns:
        tuple[str, str]: (prompt, system_prompt)
    """
    prompt_path = Path(__file__).parent / "chatbot_prompt.md"
    system_prompt = PromptLoader.read_file(str(prompt_path))
    prompt = f"## Context\n\n{context}\n\n## User question\n\n{user_query}"
    return prompt, system_prompt


class RAGChatbot:
    """
    RAG Chatbot: retrieval + LLM generation.
    Uses SearchPipeline for retrieval and the configured text model (OpenAI/Azure) for answer generation.
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
        self._llm: Optional[BaseTextModel] = None

        self._pipeline = SearchPipeline(config=config.retrieval)
        self._llm = create_text_model(
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
    def llm(self) -> BaseTextModel:
        """LLM for answer generation."""
        if self._llm is None:
            raise RuntimeError("Chatbot has been closed")
        return self._llm

    def ask(self, query: str, job_id: Optional[str] = None) -> str:
        """
        Answer a question using retrieval + LLM generation.

        Args:
            query: User question.
            job_id: Optional job identifier for tracking logs.

        Returns:
            str: Generated answer (without the machine-readable sources block).
        """
        answer, _ = self.ask_with_sources(query=query, job_id=job_id)
        return answer

    def ask_with_sources(
        self, query: str, job_id: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Like `ask`, but also returns the sources cited by the LLM.

        Returns:
            (answer, [{"file_name": str, "pages": [str, ...]}, ...])
        """
        self.logger.info("Processing query", extra={"query_length": len(query)})
        results = self.pipeline.search(query=query, job_id=job_id)
        context = _format_context(results)
        prompt, system_prompt = _get_rag_prompt(user_query=query, context=context)

        system = self.config.system_prompt if self.config.system_prompt else system_prompt
        response = self.llm.call_text_model(
            prompt=prompt,
            system_prompt=system,
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
        )
        return split_sources(response, results)

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
