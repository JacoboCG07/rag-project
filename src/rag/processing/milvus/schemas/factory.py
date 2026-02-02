"""
Factory for Milvus schemas
Implements Factory pattern for schema creation
"""

from typing import Dict, Type
from pymilvus import CollectionSchema

from .base import SchemaProvider
from .exceptions import SchemaNotFoundError
from .strategies.rag_schema import DocumentSchemaProvider


class Schemas:
    """
    Factory for Milvus schemas.
    Resolves by name and delegates to the appropriate strategy.
    Implements Factory pattern for schema creation.
    """

    _registry: Dict[str, Type[SchemaProvider]] = {
        "document": DocumentSchemaProvider,
    }

    def get_schema(self, *, name_schema: str, embedding_dim: int = 1536) -> CollectionSchema:
        """
        Returns the CollectionSchema corresponding to the given name.

        Args:
            name_schema: Schema name ('document').
            embedding_dim: Embedding vector dimension (default 1536).

        Returns:
            CollectionSchema: Schema ready to create the collection in Milvus.

        Raises:
            SchemaNotFoundError: If no provider exists for name_schema.
        """
        key = name_schema.strip().lower()
        provider_cls = self._registry.get(key)
        
        if not provider_cls:
            raise SchemaNotFoundError(
                f"No schema found for '{name_schema}'. "
                f"Valid options: {list(self._registry.keys())}"
            )
        
        provider = provider_cls(embedding_dim=embedding_dim)
        return provider.build()

