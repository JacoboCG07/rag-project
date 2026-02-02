"""
Base interface for text embedders
Implements Strategy pattern to allow different embedding providers
"""

from typing import Tuple, Optional, List
from abc import ABC, abstractmethod
from src.utils import get_logger
from functools import partial
import multiprocessing
import importlib
import time

class RateLimitError(Exception):
    """Custom exception for rate limit errors."""
    pass


class BaseEmbedder(ABC):
    """
    Base interface for text embedding generation.
    Implements Strategy pattern to allow different embedding providers.
    """

    @abstractmethod
    def generate_embedding(
        self,
        text: str
    ) -> Tuple[List[float], Optional[int]]:
        """
        Generates an embedding vector for the given text.

        Args:
            text: Text to generate embedding for.

        Returns:
            Tuple[List[float], Optional[int]]: (embedding_vector, token_count)
                - embedding_vector: List of floats representing the embedding
                - token_count: Number of tokens used (None if not available)

        Raises:
            RateLimitError: If rate limit is exceeded (should be caught and retried).
            Exception: For other errors.
        """
        pass

    @abstractmethod
    def _get_serializable_config(self) -> dict:
        """
        Returns a dictionary with serializable configuration parameters.
        Used for multiprocessing to recreate the embedder in worker processes.

        Returns:
            dict: Dictionary with configuration parameters (must be pickleable).
        """
        pass

    @classmethod
    @abstractmethod
    def _from_config(cls, config: dict) -> 'BaseEmbedder':
        """
        Creates an embedder instance from a configuration dictionary.
        Used for multiprocessing to recreate the embedder in worker processes.

        Args:
            config: Dictionary with configuration parameters.

        Returns:
            BaseEmbedder: New embedder instance.
        """
        pass

    def generate_embeddings_batch(
        self,
        *,
        texts: List[str],
        batch_size: int = 20,
        max_retries: int = 5,
        retry_delay: int = 15
    ) -> List[Tuple[List[float], Optional[int]]]:
        """
        Generates embeddings for a batch of texts using multiprocessing.
        Processes texts in batches with retry logic for rate limits.
        This implementation is shared by all embedder providers.

        Args:
            texts: List of texts to generate embeddings for.
            batch_size: Number of texts to process in each batch (default 20).
            max_retries: Maximum number of retries for rate limit errors (default 5).
            retry_delay: Delay in seconds before retrying after rate limit (default 15).

        Returns:
            List[Tuple[List[float], Optional[int]]]: List of (embedding, token_count) tuples.
                None values for invalid texts.

        Raises:
            Exception: If processing fails after all retries.
        """
        logger = get_logger(__name__)
        
        if not texts:
            logger.debug("Empty texts list provided for batch embedding generation")
            return []

        logger.debug(
            "Starting batch embedding generation",
            extra={
                "total_texts": len(texts),
                "batch_size": batch_size,
                "max_retries": max_retries,
                "retry_delay": retry_delay
            }
        )

        # Filter empty texts and keep track of original indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)

        if not valid_texts:
            logger.warning("No valid texts found after filtering")
            return []

        # Split into batches
        batches = [
            valid_texts[i:i + batch_size]
            for i in range(0, len(valid_texts), batch_size)
        ]

        logger.debug(
            "Batches created for processing",
            extra={
                "total_batches": len(batches),
                "batch_size": batch_size
            }
        )

        # Process batches with multiprocessing
        all_results = []
        with multiprocessing.Pool(processes=min(len(batches), multiprocessing.cpu_count())) as pool:
            # Get serializable config and class name
            embedder_config = self._get_serializable_config()
            embedder_class_name = self.__class__.__name__
            embedder_module = self.__class__.__module__
            
            # Create partial function with serializable parameters
            process_batch_func = partial(
                self._process_batch_with_retry,
                embedder_class_name=embedder_class_name,
                embedder_module=embedder_module,
                embedder_config=embedder_config,
                max_retries=max_retries,
                retry_delay=retry_delay
            )

            # Process batches in parallel
            batch_results = pool.map(process_batch_func, batches)

            # Flatten results
            for batch_result in batch_results:
                all_results.extend(batch_result)

        # Create result list with None for invalid texts
        results = [None] * len(texts)
        for idx, result in zip(valid_indices, all_results):
            results[idx] = result

        successful_count = sum(1 for r in results if r is not None)
        logger.info(
            "Batch embedding generation completed",
            extra={
                "total_texts": len(texts),
                "successful_embeddings": successful_count,
                "failed_embeddings": len(texts) - successful_count,
                "batches_processed": len(batches)
            }
        )

        return results

    @staticmethod
    def _process_batch_with_retry(
        batch: List[str],
        *,
        embedder_class_name: str,
        embedder_module: str,
        embedder_config: dict,
        max_retries: int,
        retry_delay: int
    ) -> List[Tuple[List[float], Optional[int]]]:
        """
        Processes a batch of texts with retry logic for rate limits.
        This method is called in separate processes.
        Creates a new embedder instance in the worker process to avoid pickle issues.

        Args:
            batch: List of texts to process.
            embedder_class_name: Name of the embedder class.
            embedder_module: Module path of the embedder class.
            embedder_config: Dictionary with embedder configuration.
            max_retries: Maximum number of retries.
            retry_delay: Delay in seconds before retrying.

        Returns:
            List[Tuple[List[float], Optional[int]]]: List of (embedding, token_count) tuples.
        """
        # Import and create embedder instance in worker process
        
        logger = get_logger(__name__)
        module = importlib.import_module(embedder_module)
        embedder_class = getattr(module, embedder_class_name)
        embedder = embedder_class._from_config(embedder_config)
        
        logger.debug(
            "Processing batch in worker process",
            extra={
                "batch_size": len(batch),
                "embedder_class": embedder_class_name
            }
        )
        
        results = []
        for text_idx, text in enumerate(batch):
            for attempt in range(1, max_retries + 1):
                try:
                    embedding, token_count = embedder.generate_embedding(text)
                    results.append((embedding, token_count))
                    break  # Success, move to next text

                except RateLimitError:
                    if attempt < max_retries:
                        logger.warning(
                            f"Rate limit error, retrying (attempt {attempt}/{max_retries})",
                            extra={
                                "attempt": attempt,
                                "max_retries": max_retries,
                                "retry_delay": retry_delay
                            }
                        )
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Max retries reached, raise error
                        error_msg = (
                            f"Failed to generate embedding after {max_retries} retries. "
                            f"Rate limit exceeded for text: {text[:50]}..."
                        )
                        logger.error(
                            error_msg,
                            extra={
                                "text_index": text_idx,
                                "max_retries": max_retries
                            }
                        )
                        raise Exception(error_msg)

                except Exception as e:
                    # Other errors, raise immediately
                    logger.error(
                        f"Error generating embedding: {str(e)}",
                        extra={
                            "text_index": text_idx,
                            "error_type": type(e).__name__
                        },
                        exc_info=True
                    )
                    raise Exception(f"Error generating embedding: {str(e)}") from e

        logger.debug(
            "Batch processed successfully in worker",
            extra={
                "batch_size": len(batch),
                "successful": len(results)
            }
        )
        
        return results

