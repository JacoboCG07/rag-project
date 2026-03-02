"""
Base interface for text chunkers.

Define un contrato común basado en DTOs para las diferentes
estrategias de chunking.
"""

from abc import ABC, abstractmethod
from typing import List

from .dto import BaseChunkDTO


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
    ) -> List[BaseChunkDTO]:
        """
        Divide una lista de textos en chunks más pequeños.

        Args:
            texts: Lista de textos a trocear (típicamente páginas).

        Returns:
            List[BaseChunkDTO]: Lista de DTOs, uno por chunk generado.
        """
        pass

