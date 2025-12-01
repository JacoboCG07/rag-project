"""
OpenAI Vision model implementation
Makes calls to OpenAI Vision API for various vision tasks
"""

from typing import Dict, Any, Optional, List, Union
import os

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base_vision_model import BaseVisionModel


class OpenAIVisionModel(BaseVisionModel):
    """
    Vision model using OpenAI Vision API.
    Supports multiple models and can be used for various vision tasks.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 500,
        temperature: float = 0.0
    ):
        """
        Initializes the OpenAI Vision model.

        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var).
            model: Vision model to use (default 'gpt-4o', also 'gpt-4-vision-preview').
            max_tokens: Default maximum tokens for responses (default 500).
            temperature: Default temperature for generation (default 0.0 for deterministic).

        Raises:
            ImportError: If openai package is not installed.
            ValueError: If api_key is not provided and OPENAI_API_KEY env var is not set.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. Install it with: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Provide it as parameter or set OPENAI_API_KEY environment variable."
            )

        self.model = model
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.client = OpenAI(api_key=self.api_key)

    def call_vision_model(
        self,
        *,
        prompt: str,
        images: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Makes a call to OpenAI Vision API with a prompt and one or more images.
        This is a generic method that can be used for various vision tasks.

        Args:
            prompt: Text prompt to send to the vision model.
            images: Single image (str) or list of images (List[str]), encoded in base64 format.
                   Can be raw base64 strings or data URLs (data:image/...;base64,...).
            max_tokens: Maximum tokens for the response (uses default if not provided).
            temperature: Temperature for generation (uses default if not provided).
            **kwargs: Additional OpenAI API parameters (e.g., top_p, frequency_penalty, etc.).

        Returns:
            str: Response text from the vision model.

        Raises:
            ValueError: If prompt or images are empty.
            Exception: If the API call fails.
        """
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        if not images:
            raise ValueError("images cannot be empty")

        # Normalize images to list
        if isinstance(images, str):
            images_list = [images]
        else:
            images_list = images

        if not images_list:
            raise ValueError("images list cannot be empty")

        try:
            # Prepare content for OpenAI API
            content = [{"type": "text", "text": prompt.strip()}]

            # Add images to content
            for image in images_list:
                image_data = self._prepare_image_data(image_base64=image)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_data
                    }
                })

            # Prepare API parameters
            api_params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
                "temperature": temperature if temperature is not None else self.default_temperature,
            }

            # Add any additional kwargs
            api_params.update(kwargs)

            # Call OpenAI Vision API
            response = self.client.chat.completions.create(**api_params)

            result = response.choices[0].message.content
            if not result:
                raise Exception("OpenAI Vision API returned an empty response.")

            return result.strip()

        except Exception as e:
            raise Exception(f"Error calling OpenAI Vision API: {str(e)}") from e

    def _prepare_image_data(
        self,
        *,
        image_base64: str
    ) -> str:
        """
        Prepares image data for OpenAI Vision API.
        Converts raw base64 to data URL format if needed.

        Args:
            image_base64: Image encoded in base64 format (raw or data URL).

        Returns:
            str: Data URL format for OpenAI API (data:image/{format};base64,{base64_string}).
        """
        # OpenAI expects data URL format: data:image/{format};base64,{base64_string}
        # If already in data URL format, return as is
        if image_base64.startswith("data:image/"):
            return image_base64
        
        # Otherwise, assume it's raw base64 and add data URL prefix
        # Try to detect format from base64 header or default to png
        # For now, default to png (most common)
        return f"data:image/png;base64,{image_base64}"

