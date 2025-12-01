"""
Base interface for text models
Implements Strategy pattern to allow different text model providers
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class BaseTextModel(ABC):
    """
    Base interface for text model calls.
    Implements Strategy pattern to allow different text model providers.
    Provides a generic method for calling text models with prompts.
    """

    @abstractmethod
    def call_text_model(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Makes a call to a text model with a prompt.
        This is a generic method that can be used for various text generation tasks.

        Args:
            prompt: Text prompt to send to the model.
            system_prompt: Optional system prompt to set the model's behavior.
            messages: Optional list of message dictionaries with 'role' and 'content' keys.
                     If provided, uses conversation format instead of single prompt.
                     Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            **kwargs: Additional parameters specific to the text model provider:
                     - max_tokens: int (optional)
                     - temperature: float (optional)
                     - top_p: float (optional)
                     - Other provider-specific parameters

        Returns:
            str: Response text from the text model.

        Raises:
            ValueError: If prompt is empty and messages is not provided.
            Exception: If the API call fails.
        """
        pass

