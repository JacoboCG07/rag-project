"""
Shared Azure OpenAI settings (endpoint, API version and client creation)
"""

import os
from typing import Optional

try:
    from openai import AzureOpenAI
except ImportError:  # handled by the model classes (OPENAI_AVAILABLE)
    AzureOpenAI = None

AZURE_API_KEY_ENV = "AZURE_OPENAI_API_KEY"
DEFAULT_API_VERSION = "2024-12-01-preview"


def azure_endpoint(value: Optional[str] = None) -> str:
    """Returns the Azure endpoint (param or AZURE_OPENAI_BASE_URL env var)."""
    endpoint = value or os.getenv("AZURE_OPENAI_BASE_URL")
    if not endpoint:
        raise ValueError("Azure endpoint is required. Set AZURE_OPENAI_BASE_URL environment variable.")
    return endpoint


def azure_api_version(value: Optional[str] = None) -> str:
    """Returns the Azure API version (param, AZURE_OPENAI_API_VERSION env var or default)."""
    return value or os.getenv("AZURE_OPENAI_API_VERSION") or DEFAULT_API_VERSION


def create_azure_client(*, api_key: str, endpoint: str, api_version: str):
    """Creates an AzureOpenAI client."""
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
