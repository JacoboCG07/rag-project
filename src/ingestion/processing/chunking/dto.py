"""
DTOs para la salida del chunker de texto.

Definen un contrato mínimo (`BaseChunkDTO`) que todo chunker debe cumplir.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ChunkMetadata:
    """
    Metadatos mínimos asociados a cada chunk de texto.
    """

    pages: List[int]
    chapters: Optional[List[str]] = None


@dataclass
class BaseChunkDTO:
    """
    DTO base que todo chunker debe devolver como mínimo.
    """

    text: str
    metadata: ChunkMetadata