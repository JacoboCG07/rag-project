"""
Chunk processor for embeddings generation
Handles embedding generation for text chunks
"""

from typing import List, Tuple, Callable, Any
from src.utils import get_logger


class ChunkProcessor:
    """
    Processes chunks to generate embeddings.
    Responsible for generating embeddings for each chunk.
    """

    def __init__(
        self,
        *,
        generate_embeddings_func: Callable[[str], Any],
    ):
        """
        Initializes the chunk processor.

        Args:
            generate_embeddings_func: Function to generate embeddings (must receive text and return embedding).
        """
        self.generate_embeddings_func = generate_embeddings_func
        self.logger = get_logger(__name__)

        self.logger.info("Initializing ChunkProcessor")

    def process(self, chunks: List[str]) -> Tuple[List[str], List[List[float]]]:
        """
        Processes chunks and generates embeddings.

        Args:
            chunks: List of text chunks.

        Returns:
            Tuple[List[str], List[List[float]]]: (chunks, embeddings).
                Only returns chunks that were successfully processed.
        """
        if not chunks:
            self.logger.debug("Empty chunks list provided, returning empty result")
            return [], []

        self.logger.debug(
            "Starting chunk processing", extra={"chunks_count": len(chunks)}
        )

        processed_chunks = []
        embeddings = []
        skipped_count = 0

        for i, chunk in enumerate(chunks):
            if not chunk or not isinstance(chunk, str):
                skipped_count += 1
                self.logger.debug(
                    f"Skipping invalid chunk at index {i}",
                    extra={"chunk_index": i},
                )
                continue

            # Clean text
            cleaned_chunk = chunk.strip()
            if not cleaned_chunk:
                skipped_count += 1
                self.logger.debug(
                    f"Skipping empty chunk at index {i}",
                    extra={"chunk_index": i},
                )
                continue

            # Generate embedding
            try:
                embedding = self.generate_embeddings_func(cleaned_chunk)
                if isinstance(embedding, tuple):
                    # If it returns (embedding, token_count), extract just the embedding
                    embedding, _ = embedding

                processed_chunks.append(cleaned_chunk)
                embeddings.append(embedding)
            except Exception as e:
                skipped_count += 1
                self.logger.warning(
                    f"Failed to generate embedding for chunk at index {i}: {str(e)}",
                    extra={
                        "chunk_index": i,
                        "error_type": type(e).__name__,
                    },
                )
                continue

        if skipped_count > 0:
            self.logger.debug(
                "Some chunks were skipped during processing",
                extra={
                    "total_chunks": len(chunks),
                    "processed_chunks": len(processed_chunks),
                    "skipped_chunks": skipped_count,
                },
            )

        self.logger.info(
            "Chunk processing completed",
            extra={
                "input_chunks": len(chunks),
                "processed_chunks": len(processed_chunks),
                "skipped_chunks": skipped_count,
            },
        )

        return processed_chunks, embeddings


