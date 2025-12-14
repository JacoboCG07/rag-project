"""
LLM-based summarizer implementation
Uses text models to generate document summaries
"""

from typing import Optional
from llms.text import BaseTextModel, OpenAITextModel
from src.utils.utils import PromptLoader


class LLMSummarizer:
    """
    Summarizer using LLM text models.
    Supports various text models (OpenAI, Anthropic, etc.) through the BaseTextModel interface.
    """

    def __init__(
        self,
        *,
        text_model: [BaseTextModel] = None,
        max_tokens: int = 1_000,
        temperature: float = 0.3
    ):
        """
        Initializes the LLM summarizer.

        Args:
            text_model: Optional BaseTextModel instance to use.
                       If None, creates an OpenAITextModel with provided parameters.
            api_key: API key for the text model (only used if text_model is None).
            model: Model name to use (only used if text_model is None, default 'gpt-4o').
            max_tokens: Maximum tokens for the summary (default 500).
            temperature: Temperature for generation (default 0.3, lower for more focused summaries).
            prompt_template: Optional custom prompt template.
                           If None, uses default template.
                           Template should contain {text} placeholder for the document text.

        Raises:
            ValueError: If both text_model and api_key are None.
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
            raise ValueError("Text must be a non-empty string")

        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty after stripping")

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
            return summary.strip()
        except Exception as e:
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

        system_prompt = PromptLoader.read_file("src/rag/processing/summarizer/summarizer_prompt.md")
        return text, system_prompt

