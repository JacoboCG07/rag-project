"""
Base interface for text chunkers
Implements Strategy pattern to allow different chunking strategies
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class BaseChunker(ABC):
    """
    Base interface for text chunking.
    Implements Strategy pattern to allow different chunking strategies.
    """

    @abstractmethod
    def chunk(
        self,
        *,
        texts: List[str],
        return_metadata: bool = False
    ) -> List[str] | Tuple[List[str], List[dict]]:
        """
        Chunks a list of texts into smaller segments.

        Args:
            texts: List of texts to chunk (typically pages).
            return_metadata: If True, returns metadata (pages, chapters) along with chunks.

        Returns:
            If return_metadata=False: List[str] - List of chunked texts.
            If return_metadata=True: Tuple[List[str], List[dict]] - (chunks, metadata_list)
                where metadata_list contains dicts with:
                - 'pages': List[int] - Pages of the chunk
                - 'chapters': str or List[str] - Chapters of the chunk
        """
        pass

