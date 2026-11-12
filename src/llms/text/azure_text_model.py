"""
Azure OpenAI Text model implementation
Same behaviour as OpenAITextModel; only the client changes.
"""

from typing import Optional

from .openai_text_model import OpenAITextModel
from src.llms.azure_config import (
    AZURE_API_KEY_ENV, azure_endpoint, azure_api_version, azure_deployment, create_azure_client,
)


class AzureTextModel(OpenAITextModel):
    """
    Text model using Azure OpenAI.
    `model` is the Azure DEPLOYMENT name (default: AZURE_OPENAI_CHAT_DEPLOYMENT_NAME).
    """

    API_KEY_ENV = AZURE_API_KEY_ENV

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        max_tokens: int = 10_000,
        temperature: float = 0.3,
        endpoint: Optional[str] = None,
    ):
        # Must be set before super().__init__(), which calls _create_client()
        self.endpoint = azure_endpoint(endpoint)
        self.api_version = azure_api_version()
        super().__init__(
            model=azure_deployment(model, "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _create_client(self):
        return create_azure_client(
            api_key=self.api_key, endpoint=self.endpoint, api_version=self.api_version
        )
