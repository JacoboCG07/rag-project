"""
Factory for text chunkers
Implements Factory pattern for chunker creation
"""

from typing import Dict, Type, Optional

from .base_chunker import BaseChunker
from .text_chunker import TextChunker


class ChunkingFactory:
    """
    Factory for text chunkers.
    Resolves by strategy name and creates appropriate chunker instance.
    Implements Factory pattern for chunker creation.
    """

    _registry: Dict[str, Type[BaseChunker]] = {
        "characters": TextChunker,
        "default": TextChunker,
    }

    @classmethod
    def create_chunker(
        cls,
        *,
        strategy: str = "default",
        chunk_size: int = 2000,
        overlap: int = 0,
        detect_chapters: bool = True,
        **kwargs
    ) -> BaseChunker:
        """
        Creates a chunker instance based on the specified strategy.

        Args:
            strategy: Chunking strategy name ('characters', 'default').
            chunk_size: Maximum size of each chunk (default 2000).
            overlap: Number of characters to overlap between chunks (default 0).
            detect_chapters: Whether to detect chapters in text (default True).
            **kwargs: Additional parameters for specific chunker implementations.

        Returns:
            BaseChunker: Chunker instance.

        Raises:
            ValueError: If strategy is not registered.
        """
        key = strategy.strip().lower()
        chunker_cls = cls._registry.get(key)

        if not chunker_cls:
            raise ValueError(
                f"No chunking strategy found for '{strategy}'. "
                f"Valid options: {list(cls._registry.keys())}"
            )

        # Create chunker instance with appropriate parameters
        if chunker_cls == TextChunker:
            return chunker_cls(
                chunk_size=chunk_size,
                overlap=overlap,
                detect_chapters=detect_chapters,
                **kwargs
            )
        else:
            # For future chunker implementations
            return chunker_cls(**kwargs)

    @classmethod
    def register_strategy(
        cls,
        *,
        name: str,
        chunker_class: Type[BaseChunker]
    ) -> None:
        """
        Registers a new chunking strategy.

        Args:
            name: Strategy name.
            chunker_class: Chunker class that implements BaseChunker.
        """
        cls._registry[name.lower()] = chunker_class

