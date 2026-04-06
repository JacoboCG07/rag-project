"""
Milvus integration for RAG: client (insert, schemas, indices), vector search, partition env helpers, insert DTOs.
"""

from .client import (
    MilvusClient,
    ConnectionManager,
    CollectionManager,
    DataManager,
    MilvusConnectionError,
    MilvusCollectionError,
    MilvusInsertError,
    SchemaProvider,
    DocumentSchemaProvider,
    Schemas,
    SchemaNotFoundError,
    IndexProvider,
    DefaultRAGIndexProvider,
    HNSWIndexProvider,
    IVF_SQ8IndexProvider,
    FLATIndexProvider,
    Indices,
    IndexNotFoundError,
)
from .searcher import MilvusSearcher
from .partitions import partition_documents, partition_summaries
from .dto import SCHEMA_KEYS, MilvusInsertDTO, build_milvus_insert_dict

__all__ = [
    "MilvusClient",
    "ConnectionManager",
    "CollectionManager",
    "DataManager",
    "MilvusConnectionError",
    "MilvusCollectionError",
    "MilvusInsertError",
    "SchemaProvider",
    "DocumentSchemaProvider",
    "Schemas",
    "SchemaNotFoundError",
    "IndexProvider",
    "DefaultRAGIndexProvider",
    "HNSWIndexProvider",
    "IVF_SQ8IndexProvider",
    "FLATIndexProvider",
    "Indices",
    "IndexNotFoundError",
    "MilvusSearcher",
    "partition_documents",
    "partition_summaries",
    "SCHEMA_KEYS",
    "MilvusInsertDTO",
    "build_milvus_insert_dict",
]
