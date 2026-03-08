"""
LLM-based summarizer implementation
Uses text models to generate document summaries
"""

from typing import Optional
from pathlib import Path
from src.llms.text import BaseTextModel, OpenAITextModel
from src.utils.utils import PromptLoader
from src.utils import get_logger

_TRUNCATION_SUFFIX = (
    "\n\n[... Texto recortado: el documento excedía el límite de caracteres. "
    "El resumen se basa en la primera parte.]"
)


def _truncate_text(text: str, max_chars: int) -> str:
    """
    Trunca el texto a max_chars caracteres, intentando cortar en límite de
    párrafo o palabra. Añade sufijo indicando el recorte.
    """
    max_content = max_chars - len(_TRUNCATION_SUFFIX)
    if len(text) <= max_content:
        return text

    # Buscar último punto o doble salto en los últimos 500 caracteres
    search_start = max(0, max_content - 500)
    search_region = text[search_start:max_content]
    cut_idx = -1
    for sep in (". ", "\n\n"):
        pos = search_region.rfind(sep)
        if pos != -1:
            cut_idx = search_start + pos + len(sep)
            break

    if cut_idx > 0:
        truncated = text[:cut_idx].rstrip()
    else:
        # Cortar en último espacio antes del límite
        truncated = text[:max_content].rstrip()
        last_space = truncated.rfind(" ")
        if last_space > max_content // 2:
            truncated = truncated[:last_space]

    return truncated + _TRUNCATION_SUFFIX


class LLMSummarizer:
    """
    Summarizer using LLM text models.
    Supports various text models (OpenAI, Anthropic, etc.) through the BaseTextModel interface.
    """

    def __init__(
        self,
        *,
        text_model: Optional[BaseTextModel] = None,
        max_tokens: int = 1_000,
        temperature: float = 0.3,
        max_input_chars: int = 100_000
    ):
        """
        Initializes the LLM summarizer.

        Args:
            text_model: BaseTextModel instance to use. Must be provided.
            max_tokens: Maximum tokens for the summary (default 1000).
            temperature: Temperature for generation (default 0.3, lower for more focused summaries).
            max_input_chars: Maximum characters of input text (default 100000). Longer text is truncated with a warning.

        Raises:
            ValueError: If text_model is not provided or is not an instance of BaseTextModel.
        """
        if not isinstance(text_model, BaseTextModel):
            raise ValueError(
                "Either 'text_model' must be provided. "
                "If using OpenAI,"
                "For other providers, provide a 'text_model' instance."
            )
        
        self.text_model: BaseTextModel = text_model
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature
        self.max_input_chars: int = max_input_chars
        self.logger = get_logger(__name__)
        
        self.logger.info(
            "Initializing LLMSummarizer",
            extra={
                "max_tokens": max_tokens,
                "temperature": temperature,
                "max_input_chars": max_input_chars,
                "text_model_type": type(text_model).__name__
            }
        )

    def generate_summary(self, text: str) -> str:
        """
        Generates a summary from the given text using an LLM.

        Args:
            text: Full text content to summarize.

        Returns:
            str: Generated summary of the text.

        Raises:
            ValueError: If text is empty or invalid.
            Exception: If summarization fails.
        """
        if not text or not isinstance(text, str):
            self.logger.error("Text must be a non-empty string")
            raise ValueError("Text must be a non-empty string")

        text = text.strip()
        if not text:
            self.logger.error("Text cannot be empty after stripping")
            raise ValueError("Text cannot be empty after stripping")

        original_len = len(text)
        if original_len > self.max_input_chars:
            self.logger.warning(
                f"Texto muy largo ({original_len} caracteres). Truncando a {self.max_input_chars} caracteres "
                "para evitar exceder el límite del modelo. El resumen se basará en la primera parte del documento.",
                extra={
                    "original_length": original_len,
                    "max_input_chars": self.max_input_chars
                }
            )
            text = _truncate_text(text, self.max_input_chars)

        self.logger.debug(
            "Starting summary generation",
            extra={
                "text_length": len(text),
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        )

        # Build prompt using template
        prompt, system_prompt = self._get_summary_prompt(text)

        # Generate summary using text model
        try:
            summary = self.text_model.call_text_model(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            self.logger.info(
                "Summary generated successfully",
                extra={
                    "original_text_length": original_len,
                    "summary_length": len(summary.strip())
                }
            )
            
            return summary.strip()
        except Exception as e:
            error_detail = str(e)
            if isinstance(e, KeyError) or error_detail == "'error'":
                error_detail = (
                    f"{error_detail} — Posible error de la API de OpenAI al parsear la respuesta. "
                    "Verifica OPENAI_API_KEY, límites de uso y conectividad."
                )
            self.logger.error(
                f"Error generating summary: {error_detail}",
                extra={
                    "text_length": original_len,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            raise Exception(f"Error generating summary: {error_detail}") from e

    @staticmethod
    def _get_summary_prompt(text: str) -> str:
        """
        Returns the default prompt template for summarization.

        Args:
            text: Full text content to summarize.

        Returns:
            str: Default prompt template.
        """

        # Obtener la ruta del archivo de prompt relativa al directorio del módulo
        # __file__ apunta a este archivo en src/ingestion/processing/summarizer/llm_summarizer.py
        # El prompt está en el mismo directorio que este archivo
        prompt_path = Path(__file__).parent / "summarizer_prompt.md"
        system_prompt = PromptLoader.read_file(str(prompt_path))
        return text, system_prompt

