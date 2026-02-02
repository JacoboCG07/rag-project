"""
Embeddings module for LLMs
Handles text embedding generation
"""

from .base_embedder import BaseEmbedder
from .openai_embedder import OpenAIEmbedder

__all__ = ['BaseEmbedder', 'OpenAIEmbedder']

