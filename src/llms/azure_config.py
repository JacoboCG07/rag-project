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


def azure_endpoint(value: Optional[str] = None) -> str:
    """Returns the Azure endpoint (param or AZURE_OPENAI_BASE_URL env var)."""
    endpoint = value or os.getenv("AZURE_OPENAI_BASE_URL")
    if not endpoint:
        raise ValueError("Azure endpoint is required. Set AZURE_OPENAI_BASE_URL environment variable.")
    return endpoint


def azure_api_version() -> str:
    """Returns the Azure API version from the AZURE_OPENAI_API_VERSION env var (no default)."""
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    if not api_version:
        raise ValueError("Azure API version is required. Set AZURE_OPENAI_API_VERSION environment variable.")
    return api_version


def azure_deployment(model: Optional[str], env_var: str) -> str:
    """Returns the deployment name (param or the given env var, no default)."""
    deployment = model or os.getenv(env_var)
    if not deployment:
        raise ValueError(f"Azure deployment name is required. Set {env_var} environment variable.")
    return deployment


def create_azure_client(*, api_key: str, endpoint: str, api_version: str):
    """Creates an AzureOpenAI client."""
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
