"""
OpenAI embedder implementation
Generates embeddings using OpenAI API
"""

from typing import Tuple, Optional, List
import os

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base_embedder import BaseEmbedder, RateLimitError
from src.utils import get_logger


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedder using OpenAI API for text embeddings.
    Supports multiple models and token counting.
    """

    # Mapping of OpenAI embedding models to their vector dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "text-embedding-2": 1536,
    }

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        count_tokens: bool = True
    ):
        """
        Initializes the OpenAI embedder.

        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var).
            model: Embedding model to use (default 'text-embedding-3-small').
            count_tokens: Whether to count tokens (default True).

        Raises:
            ImportError: If openai package is not installed.
            ValueError: If api_key is not provided and OPENAI_API_KEY env var is not set.
        """

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Provide it as parameter or set OPENAI_API_KEY environment variable."
            )

        self.model = model
        self.count_tokens = count_tokens
        self.client = OpenAI(api_key=self.api_key)
        self.logger = get_logger(__name__)
        
        self.logger.info(
            "Initializing OpenAIEmbedder",
            extra={
                "model": model,
                "count_tokens": count_tokens
            }
        )

    def generate_embedding(
        self,
        *,
        text: str
    ) -> Tuple[List[float], Optional[int]]:
        """
        Generates an embedding vector for the given text using OpenAI API.

        Args:
            text: Text to generate embedding for.

        Returns:
            Tuple[List[float], Optional[int]]: (embedding_vector, token_count)
                - embedding_vector: List of floats representing the embedding
                - token_count: Number of tokens used (None if count_tokens=False)

        Raises:
            Exception: If API call fails.
        """
        if not text or not isinstance(text, str):
            self.logger.error("Text must be a non-empty string")
            raise ValueError("text must be a non-empty string")

        try:
            self.logger.debug(
                "Generating embedding with OpenAI",
                extra={
                    "model": self.model,
                    "text_length": len(text)
                }
            )
            
            response = self.client.embeddings.create(
                model=self.model,
                input=text.strip()
            )

            embedding = response.data[0].embedding
            token_count = None

            if self.count_tokens:
                # Try to get token count from usage
                if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                    token_count = response.usage.total_tokens
                else:
                    # Fallback: estimate tokens (rough approximation: 1 token ≈ 4 characters)
                    token_count = len(text) // 4

            self.logger.debug(
                "Embedding generated successfully",
                extra={
                    "model": self.model,
                    "embedding_dimensions": len(embedding),
                    "token_count": token_count
                }
            )

            return embedding, token_count

        except Exception as e:
            error_msg = str(e)
            # Check for rate limit error (429)
            is_rate_limit = "429" in error_msg or "Too Many Requests" in error_msg or "rate_limit" in error_msg.lower()
            
            if is_rate_limit:
                self.logger.warning(
                    f"Rate limit error generating embedding: {error_msg}",
                    extra={
                        "model": self.model,
                        "error_type": "rate_limit"
                    }
                )
                raise RateLimitError(f"Rate limit exceeded: {error_msg}") from e
            else:
                self.logger.error(
                    f"Error generating embedding with OpenAI: {error_msg}",
                    extra={
                        "model": self.model,
                        "error_type": type(e).__name__
                    },
                    exc_info=True
                )
            raise Exception(f"Error generating embedding with OpenAI: {str(e)}") from e

    def get_dimensions(self) -> int:
        """
        Returns the number of dimensions of the embedding vectors generated by this embedder.
        This is needed to configure the Milvus schema correctly.

        Returns:
            int: Number of dimensions in the embedding vector.

        Raises:
            ValueError: If dimensions cannot be determined for the current model.
        """
        if self.model in self.MODEL_DIMENSIONS:
            return self.MODEL_DIMENSIONS[self.model]
        else:
            raise ValueError(
                f"Unknown model '{self.model}'. "
                f"Supported models: {list(self.MODEL_DIMENSIONS.keys())}. "
                f"If this is a new model, please add it to MODEL_DIMENSIONS mapping."
            )

    def _get_serializable_config(self) -> dict:
        """
        Returns a dictionary with serializable configuration parameters.
        Used for multiprocessing to recreate the embedder in worker processes.

        Returns:
            dict: Dictionary with configuration parameters (must be pickleable).
        """
        return {
            "api_key": self.api_key,
            "model": self.model,
            "count_tokens": self.count_tokens
        }

    @classmethod
    def _from_config(cls, config: dict) -> 'OpenAIEmbedder':
        """
        Creates an embedder instance from a configuration dictionary.
        Used for multiprocessing to recreate the embedder in worker processes.

        Args:
            config: Dictionary with configuration parameters.

        Returns:
            OpenAIEmbedder: New embedder instance.
        """
        # Create instance without calling __init__ directly
        # We'll set attributes manually to avoid re-reading env var
        instance = cls.__new__(cls)
        instance.api_key = config["api_key"]
        instance.model = config["model"]
        instance.count_tokens = config["count_tokens"]
        # Create client in worker process
        instance.client = OpenAI(api_key=instance.api_key)
        return instance