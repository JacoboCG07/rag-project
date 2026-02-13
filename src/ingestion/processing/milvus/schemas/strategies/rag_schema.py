"""
RAG schema strategy for Milvus
Schema optimized for Retrieval Augmented Generation
"""

from ..base import SchemaProvider
from pymilvus import FieldSchema, CollectionSchema, DataType

class DocumentSchemaProvider(SchemaProvider):
    """
    Strategy for RAG schema (documents + embeddings).
    Schema optimized for Retrieval Augmented Generation with support for text and images.
    """

    def __init__(self, embedding_dim: int):
        """
        Initializes the RAG schema provider.

        Args:
            embedding_dim: Embedding vector dimension
        """
        self.embedding_dim = embedding_dim

    def build(self) -> CollectionSchema:
        """
        Builds the RAG schema with typical retrieval augmented generation fields.

        Returns:
            CollectionSchema: RAG schema ready to use.
        """

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=100, is_index=True),
            FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=30, is_index=True),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=20_000),
            FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="pages", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chapters", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="image_number", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="image_number_in_page", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="full_images", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=100),
        ]

        return CollectionSchema(fields, enable_dynamic=True)