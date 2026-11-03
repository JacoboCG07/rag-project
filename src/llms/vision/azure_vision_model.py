"""
Azure OpenAI Vision model implementation
Same behaviour as OpenAIVisionModel; only the client changes.
"""

import os
from typing import Optional

from .openai_vision_model import OpenAIVisionModel
from src.llms.azure_config import (
    AZURE_API_KEY_ENV, azure_endpoint, azure_api_version, create_azure_client,
)


class AzureVisionModel(OpenAIVisionModel):
    """
    Vision model using Azure OpenAI.
    `model` is the Azure DEPLOYMENT name of a vision-capable model (default: AZURE_OPENAI_CHAT_DEPLOYMENT_NAME).
    """

    API_KEY_ENV = AZURE_API_KEY_ENV

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        # Must be set before super().__init__(), which calls _create_client()
        self.endpoint = azure_endpoint(endpoint)
        self.api_version = azure_api_version(api_version)
        super().__init__(
            api_key=api_key,
            model=model or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-5.4-mini"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _create_client(self):
        return create_azure_client(
            api_key=self.api_key, endpoint=self.endpoint, api_version=self.api_version
        )
