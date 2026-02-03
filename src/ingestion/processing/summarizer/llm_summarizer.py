"""
LLM-based summarizer implementation
Uses text models to generate document summaries
"""

from typing import Optional
from pathlib import Path
from src.llms.text import BaseTextModel, OpenAITextModel
from src.utils.utils import PromptLoader
from src.utils import get_logger


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
        temperature: float = 0.3
    ):
        """
        Initializes the LLM summarizer.

        Args:
            text_model: BaseTextModel instance to use. Must be provided.
            max_tokens: Maximum tokens for the summary (default 1000).
            temperature: Temperature for generation (default 0.3, lower for more focused summaries).

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
        self.logger = get_logger(__name__)
        
        self.logger.info(
            "Initializing LLMSummarizer",
            extra={
                "max_tokens": max_tokens,
                "temperature": temperature,
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
                    "original_text_length": len(text),
                    "summary_length": len(summary.strip())
                }
            )
            
            return summary.strip()
        except Exception as e:
            self.logger.error(
                f"Error generating summary: {str(e)}",
                extra={
                    "text_length": len(text),
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            raise Exception(f"Error generating summary: {str(e)}") from e

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

