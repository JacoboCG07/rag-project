"""
Base interface for vision models
Implements Strategy pattern to allow different vision providers
"""

from abc import ABC, abstractmethod
from typing import List, Union


class BaseVisionModel(ABC):
    """
    Base interface for vision model calls.
    Implements Strategy pattern to allow different vision providers.
    Provides a generic method for calling vision models with prompts and images.
    """

    @abstractmethod
    def call_vision_model(
        self,
        *,
        prompt: str,
        images: Union[str, List[str]],
        **kwargs
    ) -> str:
        """
        Makes a call to a vision model with a prompt and one or more images.
        This is a generic method that can be used for various vision tasks.

        Args:
            prompt: Text prompt to send to the vision model.
            images: Single image (str) or list of images (List[str]), encoded in base64 format.
                   Can be raw base64 strings or data URLs (data:image/...;base64,...).
            **kwargs: Additional parameters specific to the vision provider:
                     - max_tokens: int (optional)
                     - temperature: float (optional)
                     - Other provider-specific parameters

        Returns:
            str: Response text from the vision model.

        Raises:
            ValueError: If prompt or images are empty.
            Exception: If the API call fails.
        """
        pass

