"""
Text processor for document chunking
Handles text chunking and chapter detection
"""

from typing import List, Dict, Tuple, Any
from ..chunking import ChunkingFactory
from src.utils import get_logger


class TextProcessor:
    """
    Processes text pages into chunks with metadata.
    Responsible for chunking text and detecting chapters.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 2000,
        chunk_overlap: int = 0,
        detect_chapters: bool = True,
    ):
        """
        Initializes the text processor.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
            detect_chapters: Whether to detect chapters in text.
        """
        self.logger = get_logger(__name__)
        self.chunker = ChunkingFactory.create_chunker(
            strategy="default",
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            detect_chapters=detect_chapters,
        )

        self.logger.info(
            "Initializing TextProcessor",
            extra={
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "detect_chapters": detect_chapters,
            },
        )

    def process(self, pages: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Processes pages into chunks with metadata.

        Args:
            pages: List of text pages (strings).

        Returns:
            Tuple[List[str], List[Dict]]: (chunks, chunks_metadata)
                where chunks_metadata contains dicts with 'pages' and 'chapters' keys.
        """
        if not pages:
            self.logger.debug("Empty pages list provided, returning empty result")
            return [], []

        self.logger.debug(
            "Starting text processing",
            extra={
                "pages_count": len(pages),
                "chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.overlap,
                "detect_chapters": self.chunker.detect_chapters,
            },
        )

        # Chunk pages with metadata
        chunks, metadata_list = self.chunker.chunk(texts=pages, return_metadata=True)

        self.logger.info(
            "Text processing completed",
            extra={
                "input_pages": len(pages),
                "output_chunks": len(chunks),
                "chapters_detected": sum(
                    1 for m in metadata_list if m.get("chapters")
                ),
            },
        )

        return chunks, metadata_list


