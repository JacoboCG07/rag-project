"""
Main Milvus client
Implements Facade pattern to simplify usage
"""

from typing import Optional, List, Dict, Any
from pymilvus import Collection, connections
from pymilvus.exceptions import ConnectionNotExistException

from .connection_manager import ConnectionManager
from .collection_manager import CollectionManager
from .data_manager import DataManager
from src.utils import get_logger


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
        self.logger = get_logger(__name__)

        self._collection: Optional[Collection] = None

        self.logger.info(
            "Initializing MilvusClient",
            extra={
                "dbname": dbname,
                "collection_name": collection_name,
                "alias": alias,
                "name_schema": name_schema,
                "embedding_dim": embedding_dim,
                "name_index": name_index
            }
        )

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
            self.logger.debug("Loading collection", extra={"collection_name": self.collection_name})
            self._collection = self._collection_manager.load_collection(
                collection_name=self.collection_name,
                name_schema=self.name_schema,
                embedding_dim=self.embedding_dim,
                name_index=self.name_index
            )
            self.logger.info("Collection loaded", extra={"collection_name": self.collection_name})
        return self._collection

    def create_partition(self, *, partition_name: str) -> None:
        """
        Creates a partition in the collection.

        Args:
            partition_name: Partition name.
        """
        self.logger.debug("Creating partition", extra={"partition_name": partition_name})
        collection = self.load_collection()
        self._collection_manager.create_partition(
            collection=collection,
            partition_name=partition_name
        )
        self.logger.info("Partition created", extra={"partition_name": partition_name})

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
        self.logger.debug(
            "Inserting documents",
            extra={
                "texts_count": len(texts),
                "partition_name": partition_name,
                "collection_name": self.collection_name
            }
        )
        
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
        
        self.logger.info(
            "Documents inserted successfully",
            extra={
                "documents_count": len(texts),
                "partition_name": partition_name,
                "collection_name": self.collection_name
            }
        )

    def insert_prepared_data(
        self,
        *,
        prepared_data: List[Dict[str, Any]],
        partition_name: Optional[str] = None
    ) -> None:
        """
        Inserts already prepared data into Milvus.
        Use this when data has been prepared by DocumentPreparer.

        Args:
            prepared_data: List of dictionaries with all fields ready for Milvus insertion.
            partition_name: Partition name (optional).
        """
        self.logger.debug(
            "Inserting prepared data",
            extra={
                "data_count": len(prepared_data),
                "partition_name": partition_name,
                "collection_name": self.collection_name
            }
        )

        if not prepared_data:
            self.logger.warning("No prepared data to insert")
            return

        collection = self.load_collection()
        self._data_manager.insert_data(
            collection=collection,
            data=prepared_data,
            partition_name=partition_name
        )
        
        self.logger.info(
            "Prepared data inserted successfully",
            extra={
                "documents_count": len(prepared_data),
                "partition_name": partition_name,
                "collection_name": self.collection_name
            }
        )

    def close(self) -> None:
        """Closes connection and releases resources."""
        self.logger.debug("Closing MilvusClient", extra={"collection_name": self.collection_name})
        
        # Try to release collection only if connection exists
        if self._collection:
            try:
                # Check if connection exists before releasing collection
                if connections.has_connection(self.alias):
                    self._collection_manager.release_collection(collection=self._collection)
                    self.logger.debug("Collection released", extra={"collection_name": self.collection_name})
            except ConnectionNotExistException:
                # Connection already closed, skip collection release
                self.logger.debug("Connection already closed, skipping collection release")
                pass
            except Exception as e:
                # Other errors during release are not critical, continue with disconnect
                self.logger.warning(
                    f"Error releasing collection (non-critical): {str(e)}",
                    extra={"collection_name": self.collection_name}
                )
                pass
        
        # Disconnect connection (this handles the case where connection doesn't exist)
        try:
            self._connection_manager.disconnect(alias=self.alias)
            self.logger.info("MilvusClient closed successfully", extra={"collection_name": self.collection_name})
        except Exception as e:
            # Connection may already be closed, ignore error
            self.logger.debug(f"Connection already closed: {str(e)}")
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
