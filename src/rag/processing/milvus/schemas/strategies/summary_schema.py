"""
RAG schema strategy for Milvus
Schema optimized for Retrieval Augmented Generation
"""

from pymilvus import FieldSchema, CollectionSchema, DataType
from ..base import SchemaProvider


class SummarySchemaProvider(SchemaProvider):
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
        Completar la documentación

        Returns:
            CollectionSchema: RAG schema ready to use.
        """
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=20_000),
            FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=100, is_index=True),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="type_file", dtype=DataType.VARCHAR, max_length=30, is_index=True),
            FieldSchema(name="total_pages", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="total_chapters", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="total_num_image", dtype=DataType.VARCHAR, max_length=100),
        ]
        return CollectionSchema(fields, enable_dynamic=True)