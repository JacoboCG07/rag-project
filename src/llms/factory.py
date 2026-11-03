"""
LLM provider factory.
Chooses OpenAI or Azure OpenAI classes from the LLM_PROVIDER env var ("openai" | "azure").
"""

import os

from .text import BaseTextModel, OpenAITextModel, AzureTextModel
from .vision import BaseVisionModel, OpenAIVisionModel, AzureVisionModel
from .embeddings import BaseEmbedder, OpenAIEmbedder, AzureEmbedder

PROVIDERS = ("openai", "azure")


def get_provider() -> str:
    """Returns the configured provider (LLM_PROVIDER env var, default 'openai')."""
    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Supported: {PROVIDERS}")
    return provider


def _drop_none(kwargs: dict) -> dict:
    """Removes None values so each class applies its own default (e.g. model)."""
    return {k: v for k, v in kwargs.items() if v is not None}


def create_text_model(**kwargs) -> BaseTextModel:
    """
    Creates the text model for the configured provider. kwargs: model, max_tokens, temperature.
    model=None -> provider default (OpenAI: gpt-4o, Azure: AZURE_OPENAI_CHAT_DEPLOYMENT_NAME).
    """
    kwargs = _drop_none(kwargs)
    cls = AzureTextModel if get_provider() == "azure" else OpenAITextModel
    return cls(**kwargs)


def create_vision_model(**kwargs) -> BaseVisionModel:
    """Creates the vision model for the configured provider (same defaults as create_text_model)."""
    kwargs = _drop_none(kwargs)
    cls = AzureVisionModel if get_provider() == "azure" else OpenAIVisionModel
    return cls(**kwargs)


def create_embedder(*, model: str | None = None, **kwargs) -> BaseEmbedder:
    """
    Creates the embedder for the configured provider. kwargs: count_tokens.
    If model is None, both providers use text-embedding-3-small
    (Azure: AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME if set).
    """
    kwargs = _drop_none({"model": model, **kwargs})
    cls = AzureEmbedder if get_provider() == "azure" else OpenAIEmbedder
    return cls(**kwargs)
