"""
Main Milvus client
Implements Facade pattern to simplify usage
"""

from typing import Optional, List, Dict, Any
from pymilvus import Collection

from .connection_manager import ConnectionManager
from .collection_manager import CollectionManager
from .data_manager import DataManager


class MilvusClient:
    """
    Main Milvus client that coordinates all operations.
    Implements Facade pattern to simplify usage.
    """

    def __init__(
        self,
        *,
        dbname: str,
        collection_name: str,
        alias: str = "default",
        name_schema: str = "rag",
        embedding_dim: int = 1536,
        name_index: str = "default",
        uri: Optional[str] = None,
        token: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None
    ):
        """
        Initializes the Milvus client.

        Args:
            dbname: Database name.
            collection_name: Collection name.
            alias: Connection alias.
            name_schema: Schema name to use.
            embedding_dim: Embedding vector dimension.
            name_index: Index name to use (default 'default').
            uri: Connection URI (optional).
            token: Authentication token (optional).
            host: Milvus host (optional).
            port: Milvus port (optional).
        """
        self.dbname = dbname
        self.collection_name = collection_name
        self.alias = alias
        self.name_schema = name_schema
        self.embedding_dim = embedding_dim
        self.name_index = name_index

        self._connection_manager = ConnectionManager()
        self._collection_manager = CollectionManager(alias=alias)
        self._data_manager = DataManager()

        self._collection: Optional[Collection] = None

        # Connect and load database
        self._connection_manager.connect(
            alias=alias,
            uri=uri,
            token=token,
            host=host,
            port=port
        )
        self._connection_manager.load_database(dbname=dbname, alias=alias)

    def load_collection(self) -> Collection:
        """
        Loads the collection (creates it if it doesn't exist).

        Returns:
            Collection: Loaded collection.
        """
        if self._collection is None:
            self._collection = self._collection_manager.load_collection(
                collection_name=self.collection_name,
                name_schema=self.name_schema,
                embedding_dim=self.embedding_dim,
                name_index=self.name_index
            )
        return self._collection

    def create_partition(self, *, partition_name: str) -> None:
        """
        Creates a partition in the collection.

        Args:
            partition_name: Partition name.
        """
        collection = self.load_collection()
        self._collection_manager.create_partition(
            collection=collection,
            partition_name=partition_name
        )

    def insert_documents(
        self,
        *,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None,
        partition_name: Optional[str] = None
    ) -> None:
        """
        Inserts documents into Milvus.

        Args:
            texts: List of texts to insert.
            embeddings: List of corresponding embeddings.
            metadata: Additional metadata (optional).
            partition_name: Partition name (optional).
        """
        data = self._data_manager.prepare_data_for_insertion(
            texts=texts,
            embeddings=embeddings,
            metadata=metadata
        )

        collection = self.load_collection()
        self._data_manager.insert_data(
            collection=collection,
            data=data,
            partition_name=partition_name
        )

    def close(self) -> None:
        """Closes connection and releases resources."""
        if self._collection:
            self._collection_manager.release_collection(collection=self._collection)
        self._connection_manager.disconnect(alias=self.alias)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
