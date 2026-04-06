"""
Base classes for Milvus schemas
Implements Strategy pattern for schema providers
"""

from abc import ABC, abstractmethod
from pymilvus import CollectionSchema


class SchemaProvider(ABC):
    """
    Base interface for building Milvus schemas.
    Implements Strategy pattern to allow different schema types.
    """

    @abstractmethod
    def build(self) -> CollectionSchema:
        """
        Builds and returns a Milvus CollectionSchema.

        Returns:
            CollectionSchema: Schema ready to use in collection creation.
        """
        pass