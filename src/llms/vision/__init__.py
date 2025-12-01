"""
Vision models module for RAG
Handles image description generation using vision models
"""

from .base_vision_model import BaseVisionModel
from .openai_vision_model import OpenAIVisionModel

__all__ = ['BaseVisionModel', 'OpenAIVisionModel']

