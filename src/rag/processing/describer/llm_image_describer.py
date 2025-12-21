"""
LLM-based image describer implementation
Uses vision models to generate image descriptions
"""

from typing import Optional, Union, List
from llms.vision import BaseVisionModel
from src.utils.utils import PromptLoader


class LLMImageDescriber:
    """
    Image describer using LLM vision models.
    Supports various vision models (OpenAI, etc.) through the BaseVisionModel interface.
    """

    def __init__(
        self,
        *,
        vision_model: Optional[BaseVisionModel] = None,
        max_tokens: int = 1000,
        temperature: float = 0.3
    ):
        """
        Initializes the LLM image describer.

        Args:
            vision_model: BaseVisionModel instance to use. Must be provided.
            max_tokens: Maximum tokens for the description (default 500).
            temperature: Temperature for generation (default 0.3, lower for more focused descriptions).

        Raises:
            ValueError: If vision_model is not provided or is not an instance of BaseVisionModel.
        """
        if not isinstance(vision_model, BaseVisionModel):
            raise ValueError(
                "vision_model must be provided and must be an instance of BaseVisionModel. "
                "If using OpenAI, provide an OpenAIVisionModel instance."
            )
        
        self.vision_model: BaseVisionModel = vision_model
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature

    def describe_image(
        self,
        *,
        image: Union[str, List[str]],
        prompt: Optional[str] = None
    ) -> str:
        """
        Generates a description of the given image(s) using a vision model.

        Args:
            image: Single image (str) or list of images (List[str]), encoded in base64 format.
                   Can be raw base64 strings or data URLs (data:image/...;base64,...).
            prompt: Optional custom prompt. If not provided, uses default description prompt.

        Returns:
            str: Generated description of the image(s).

        Raises:
            ValueError: If image is empty or invalid.
            Exception: If description generation fails.
        """
        if not image:
            raise ValueError("Image must be provided and non-empty")

        # Normalize to list for validation
        if isinstance(image, str):
            images_list = [image]
        else:
            images_list = image

        if not images_list or not any(images_list):
            raise ValueError("Image(s) cannot be empty")

        # Build prompt using template if not provided
        if prompt is None:
            prompt = self._get_description_prompt()

        # Generate description using vision model
        try:
            description = self.vision_model.call_vision_model(
                prompt=prompt,
                images=image,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return description.strip()
        except Exception as e:
            raise Exception(f"Error generating image description: {str(e)}") from e

    @staticmethod
    def _get_description_prompt() -> str:
        """
        Returns the default prompt template for image description.

        Returns:
            str: Default prompt template for describing images.
        """
        system_prompt = PromptLoader.read_file("src/rag/processing/describer/image_describer_prompt.md")
        return system_prompt
