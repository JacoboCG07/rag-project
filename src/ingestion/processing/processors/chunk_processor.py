"""
Chunk processor for embeddings generation
Handles embedding generation for text chunks
"""

from typing import List, Tuple, Callable, Any, Dict, Optional
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

    def process_to_embeddings(
        self,
        chunks: List[str],
        chunks_metadata: Optional[List[Dict[str, Any]]] = None,
        max_acceptable_loss: float = 0.10,
    ) -> Tuple[List[str], List[List[float]], List[Dict[str, Any]]]:
        """
        Generates embeddings for chunks. Expects chunks already cleaned (e.g. from TextProcessor).
        Per-chunk failures are discarded; if failures exceed max_acceptable_loss, raises and stops (early exit).

        Args:
            chunks: List of text chunks (already cleaned).
            chunks_metadata: Optional metadata per chunk; must have same length as chunks if provided.
            max_acceptable_loss: Max fraction of chunks that may fail (e.g. 0.10 = 10%). If exceeded, raises.

        Returns:
            Tuple of (chunks_ok, embeddings, metadata_ok), all same length. metadata_ok is empty list if chunks_metadata not provided.
        """
        if not chunks:
            self.logger.debug("Empty chunks list provided, returning empty result")
            return [], [], []

        total = len(chunks)
        max_acceptable_failures = int(max_acceptable_loss * total)
        if chunks_metadata is not None and len(chunks_metadata) != total:
            raise ValueError(
                f"chunks_metadata length ({len(chunks_metadata)}) must match chunks length ({total})"
            )

        self.logger.debug(
            "Starting chunk processing",
            extra={"chunks_count": total, "max_acceptable_loss": max_acceptable_loss},
        )

        chunks_ok: List[str] = []
        embeddings: List[List[float]] = []
        metadata_ok: List[Dict[str, Any]] = []
        failed_count = 0

        for i, chunk in enumerate(chunks):
            try:
                embedding = self.generate_embeddings_func(chunk)
                if isinstance(embedding, tuple):
                    embedding, _ = embedding

                chunks_ok.append(chunk)
                embeddings.append(embedding)
                if chunks_metadata is not None:
                    metadata_ok.append(chunks_metadata[i])
            except Exception as e:
                failed_count += 1
                self.logger.warning(
                    f"Failed to generate embedding for chunk at index {i}: {str(e)}",
                    extra={
                        "chunk_index": i,
                        "error_type": type(e).__name__,
                    },
                )
                if failed_count > max_acceptable_failures:
                    raise RuntimeError(
                        f"Embedding failures ({failed_count}) exceed max acceptable loss "
                        f"({max_acceptable_loss:.0%}, max {max_acceptable_failures} failures for {total} chunks)"
                    ) from e

        self.logger.info(
            "Chunk processing completed",
            extra={
                "input_chunks": total,
                "processed_chunks": len(chunks_ok),
                "failed_chunks": failed_count,
            },
        )

        return chunks_ok, embeddings, metadata_ok