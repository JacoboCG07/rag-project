"""
Azure OpenAI embedder implementation
Same behaviour as OpenAIEmbedder; only the client changes.
"""

import os
from typing import Optional

from .openai_embedder import OpenAIEmbedder
from src.llms.azure_config import (
    AZURE_API_KEY_ENV, azure_endpoint, azure_api_version, create_azure_client,
)


class AzureEmbedder(OpenAIEmbedder):
    """
    Embedder using Azure OpenAI.
    `model` is the Azure DEPLOYMENT name. It must match a key of MODEL_DIMENSIONS
    (e.g. "text-embedding-3-small") so the vector dimension is known.
    """

    API_KEY_ENV = AZURE_API_KEY_ENV

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        count_tokens: bool = True,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        # Must be set before super().__init__(), which calls _create_client()
        self.endpoint = azure_endpoint(endpoint)
        self.api_version = azure_api_version(api_version)
        super().__init__(
            model=model or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME", "text-embedding-3-small"),
            count_tokens=count_tokens,
        )

    def _create_client(self):
        return create_azure_client(
            api_key=self.api_key, endpoint=self.endpoint, api_version=self.api_version
        )

    def _get_serializable_config(self) -> dict:
        return {
            **super()._get_serializable_config(),
            "endpoint": self.endpoint,
            "api_version": self.api_version,
        }
