"""
Module for searching documents in Milvus.
"""
from typing import List, Dict, Optional
import sys
import os

# Add libraries-main path if it exists
libraries_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "libraries-main")
if os.path.exists(libraries_path):
    sys.path.insert(0, libraries_path)

try:
    from firemilvus import controlMilvus
except ImportError:
    controlMilvus = None

from pymilvus import Collection, connections, db, utility


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
        """Connects to Milvus and loads the collection."""
        if controlMilvus:
            self.collection = controlMilvus.load_db_and_collection(
                dbname=self.db_name,
                collection_name=self.collection_name,
                alias=self.alias
            )
        else:
            # Alternative implementation using pymilvus directly
            from dotenv import load_dotenv
            load_dotenv()
            
            host = os.getenv("MILVUS_HOST", "localhost")
            port = os.getenv("MILVUS_PORT", "19530")
            
            connections.connect(host=host, port=port, alias=self.alias)
            
            # List databases and create if it doesn't exist
            dbs = db.list_database(using=self.alias)
            if self.db_name not in dbs:
                db.create_database(self.db_name, using=self.alias)
            db.using_database(self.db_name, using=self.alias)
            
            self.collection = Collection(self.collection_name, using=self.alias)
    
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
            query_embedding: Query embedding.
            limit: Maximum number of results.
            partition_names: List of partitions to search (None = all).
            filter_expr: Optional filter expression (e.g. 'file_id == "123"').

        Returns:
            List of documents found with their scores.
        """
        try:
            # Prepare search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Execute search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="text_embedding",
                param=search_params,
                limit=limit,
                partition_names=partition_names,
                expr=filter_expr,
                output_fields=["text", "file_id", "file_name", "file_type", "pages", "chapters"]
            )
            
            # Process results
            documents = []
            if results:
                for hit in results[0]:
                    doc = {
                        "id": hit.id,
                        "score": hit.score,
                        "text": hit.entity.get("text", ""),
                        "file_id": hit.entity.get("file_id", ""),
                        "file_name": hit.entity.get("file_name", ""),
                        "source_id": hit.entity.get("file_id", ""),  # file_id used as source_id for SearchResult
                        "pages": hit.entity.get("pages", ""),
                        "chapters": hit.entity.get("chapters", ""),
                        "type_file": hit.entity.get("file_type", ""),
                    }
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            raise Exception(f"Error searching in Milvus: {str(e)}")
    
    def search_by_partition(
        self,
        query_embedding: List[float],
        partition_name: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Searches for documents in a specific partition.

        Args:
            query_embedding: Query embedding.
            partition_name: Partition name.
            limit: Maximum number of results.

        Returns:
            List of documents found.
        """
        return self.search(
            query_embedding=query_embedding,
            limit=limit,
            partition_names=[partition_name]
        )
    
    def get_partitions(self) -> List[str]:
        """
        Gets the list of partitions available in the collection.

        Returns:
            List of partition names.
        """
        try:
            partitions = self.collection.partitions
            # Filter out the default partition
            partition_names = [p.name for p in partitions if p.name != "_default"]
            return partition_names
        except Exception as e:
            raise Exception(f"Error getting partitions: {str(e)}")
    
    def disconnect(self):
        """Closes the connection to Milvus."""
        if controlMilvus:
            controlMilvus.finish_conection_and_release_collection(
                alias=self.alias,
                collection=self.collection
            )
        else:
            self.collection.release()
            connections.disconnect(alias=self.alias)

