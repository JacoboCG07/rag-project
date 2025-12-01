"""
OpenAI Text model implementation
Makes calls to OpenAI Text API for various text generation tasks
"""

from typing import Optional, List, Dict, Any
import os

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base_text_model import BaseTextModel


class OpenAITextModel(BaseTextModel):
    """
    Text model using OpenAI API.
    Supports multiple models and can be used for various text generation tasks.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        max_tokens: int = 10_000,
        temperature: float = 0.3
    ):
        """
        Initializes the OpenAI Text model.

        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var).
            model: Text model to use (default 'gpt-4o', also 'gpt-3.5-turbo', 'gpt-4', etc.).
            max_tokens: Default maximum tokens for responses (default 1000).
            temperature: Default temperature for generation (default 0.7).

        Raises:
            ImportError: If openai package is not installed.
            ValueError: If api_key is not provided and OPENAI_API_KEY env var is not set.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. Install it with: pip install openai"
            )

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Provide it as parameter or set OPENAI_API_KEY environment variable."
            )

        self.model = model
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.client = OpenAI(api_key=self.api_key)

    def call_text_model(
        self,
        *,
        prompt: str = "",
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Makes a call to OpenAI Text API with a prompt or messages.
        This is a generic method that can be used for various text generation tasks.

        Args:
            prompt: Text prompt to send to the model (used if messages is not provided).
            system_prompt: Optional system prompt to set the model's behavior.
            messages: Optional list of message dictionaries with 'role' and 'content' keys.
                     If provided, uses conversation format instead of single prompt.
                     Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            max_tokens: Maximum tokens for the response (uses default if not provided).
            temperature: Temperature for generation (uses default if not provided).
            **kwargs: Additional OpenAI API parameters (e.g., top_p, frequency_penalty, etc.).

        Returns:
            str: Response text from the text model.

        Raises:
            ValueError: If prompt is empty and messages is not provided.
            Exception: If the API call fails.
        """
        # Validate input
        if not messages and (not prompt or not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError("Either 'prompt' or 'messages' must be provided and non-empty")

        try:
            # Build messages list
            messages_list = []

            # Add system prompt if provided
            if system_prompt:
                messages_list.append({
                    "role": "system",
                    "content": system_prompt
                })

            # Use provided messages or create from prompt
            if messages:
                # Validate messages format
                for msg in messages:
                    if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                        raise ValueError(
                            "Each message must be a dict with 'role' and 'content' keys"
                        )
                messages_list.extend(messages)
            else:
                # Use single prompt
                messages_list.append({
                    "role": "user",
                    "content": prompt.strip()
                })

            # Prepare API parameters
            api_params = {
                "model": self.model,
                "messages": messages_list,
                "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
                "temperature": temperature if temperature is not None else self.default_temperature,
            }

            # Add any additional kwargs
            api_params.update(kwargs)

            # Call OpenAI Text API
            response = self.client.chat.completions.create(**api_params)

            result = response.choices[0].message.content
            if not result:
                raise Exception("OpenAI Text API returned an empty response.")

            return result.strip()

        except Exception as e:
            raise Exception(f"Error calling OpenAI Text API: {str(e)}") from e

