"""
Module for searching documents in Milvus.
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
from pymilvus import Collection, connections, db

load_dotenv()

class MilvusSearcher:
    """Class for searching documents in Milvus."""
    
    def __init__(self, db_name: str, collection_name: str, alias: str = "default"):
        """
        Initializes the Milvus searcher.

        Args:
            db_name: Database name.
            collection_name: Collection name.
            alias: Connection alias.
        """
        self.db_name = db_name
        self.collection_name = collection_name
        self.alias = alias
        self.collection = None
    
    def connect(self):
        """Connects to Milvus and loads the collection. Idempotent: if already connected and the collection is loaded, does nothing."""
        if self.collection is not None:
            return
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")

        connections.connect(host=host, port=port, alias=self.alias)

        dbs = db.list_database(using=self.alias)
        if self.db_name not in dbs:
            db.create_database(self.db_name, using=self.alias)
        db.using_database(self.db_name, using=self.alias)

        self.collection = Collection(self.collection_name, using=self.alias)
        self.collection.load()  # Load into memory to enable searches

    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        partition_names: Optional[List[str]] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict]:
        """
        Searches for similar documents in Milvus.

        Args:
            query_embedding: Query embedding vector.
            limit: Maximum number of results.
            partition_names: Partitions to search (None = all).
            filter_expr: Optional filter expression (e.g. 'file_id == "123"').

        Returns:
            List of matching documents with their scores.
        """
        try:
            # Build search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Run search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="text_embedding",
                param=search_params,
                limit=limit,
                partition_names=partition_names,
                expr=filter_expr,
                output_fields=["text", "file_id", "file_name", "file_type", "pages", "chapters"]
            )
            
            # Process results (pymilvus 2.4+: Hit.entity.get(key) has no default argument)
            def _field(entity, name: str) -> str:
                try:
                    val = entity.get(name) if hasattr(entity, "get") else getattr(entity, name, None)
                    return val if val is not None else ""
                except (TypeError, AttributeError):
                    return ""

            documents = []
            if results:
                for hit in results[0]:
                    entity = getattr(hit, "entity", hit)
                    file_id = _field(entity, "file_id")
                    doc = {
                        "id": hit.id,
                        "score": hit.score,
                        "text": _field(entity, "text"),
                        "file_id": file_id,
                        "file_name": _field(entity, "file_name"),
                        "source_id": file_id,
                        "pages": _field(entity, "pages"),
                        "chapters": _field(entity, "chapters"),
                        "file_type": _field(entity, "file_type"),
                    }
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            raise Exception(f"Error searching Milvus: {str(e)}")
    
    def search_by_partition(
        self,
        query_embedding: List[float],
        partition_name: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Searches documents in a specific partition.

        Args:
            query_embedding: Query embedding vector.
            partition_name: Partition name.
            limit: Maximum number of results.

        Returns:
            List of matching documents.
        """
        return self.search(
            query_embedding=query_embedding,
            limit=limit,
            partition_names=[partition_name]
        )
    
    def get_partitions(self) -> List[str]:
        """
        Returns the list of partitions available on the collection.

        Returns:
            List of partition names.
        """
        try:
            partitions = self.collection.partitions
            # Exclude the default partition
            partition_names = [p.name for p in partitions if p.name != "_default"]
            return partition_names
        except Exception as e:
            raise Exception(f"Error listing partitions: {str(e)}")
    
    def disconnect(self):
        """
        Releases the collection from memory. Does not close the global connection (alias), because
        in selection-based strategies (DocumentSelector) SummaryRetriever shares
        the same connection and must keep using it for subsequent queries.
        The connection is closed when the pipeline/strategy shuts down.
        """
        if self.collection is not None:
            self.collection.release()
            self.collection = None
        # Do not call connections.disconnect(): connection is shared with SummaryRetriever
