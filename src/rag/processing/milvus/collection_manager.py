"""
Milvus collection manager
Single responsibility: collection, partition and index management
"""

from typing import Optional
from pymilvus import Collection, utility
from pymilvus.exceptions import ConnectionNotExistException

from .exceptions import MilvusCollectionError
from .schemas import Schemas, SchemaNotFoundError
from .indices import Indices, IndexNotFoundError
from src.utils import get_logger


class CollectionManager:
    """
    Manages Milvus collections.
    Single responsibility: collection and partition handling.
    """

    def __init__(self, alias: str = "default"):
        """
        Initializes the collection manager.

        Args:
            alias: Connection alias to use.
        """
        self.alias = alias
        self._schemas = Schemas()
        self._indices = Indices()
        self.logger = get_logger(__name__)
        
        self.logger.debug("Initializing CollectionManager", extra={"alias": alias})

    def create_collection(
        self,
        *,
        collection_name: str,
        name_schema: str = "rag",
        embedding_dim: int = 1536
    ) -> Collection:
        """
        Creates a new collection in Milvus.

        Args:
            collection_name: Collection name.
            name_schema: Schema name to use.
            embedding_dim: Embedding vector dimension.

        Returns:
            Collection: Created collection.

        Raises:
            MilvusCollectionError: If collection cannot be created.
        """
        try:
            self.logger.debug(
                "Creating collection",
                extra={
                    "collection_name": collection_name,
                    "name_schema": name_schema,
                    "embedding_dim": embedding_dim
                }
            )
            schema = self._schemas.get_schema(name_schema=name_schema, embedding_dim=embedding_dim)
            collection = Collection(
                name=collection_name,
                schema=schema,
                using=self.alias
            )
            self.logger.info("Collection created successfully", extra={"collection_name": collection_name})
            return collection
        except SchemaNotFoundError as e:
            self.logger.error(
                f"Schema not found: {str(e)}",
                extra={"collection_name": collection_name, "name_schema": name_schema},
                exc_info=True
            )
            raise MilvusCollectionError(str(e)) from e
        except Exception as e:
            self.logger.error(
                f"Error creating collection: {str(e)}",
                extra={"collection_name": collection_name, "error_type": type(e).__name__},
                exc_info=True
            )
            raise MilvusCollectionError(
                f"Error creating collection '{collection_name}': {str(e)}"
            ) from e

    def load_collection(
        self,
        *,
        collection_name: str,
        name_schema: str = "rag",
        embedding_dim: int = 1536,
        name_index: str = "default"
    ) -> Collection:
        """
        Loads an existing collection or creates it if it doesn't exist.

        Args:
            collection_name: Collection name.
            name_schema: Schema name to use if creating.
            embedding_dim: Embedding vector dimension.
            name_index: Index name to use if creating (default 'default').

        Returns:
            Collection: Loaded or created collection.
        """
        try:
            self.logger.debug(
                "Loading collection",
                extra={
                    "collection_name": collection_name,
                    "name_schema": name_schema,
                    "embedding_dim": embedding_dim,
                    "name_index": name_index
                }
            )
            collections = utility.list_collections(using=self.alias)
            
            if collection_name not in collections:
                self.logger.info("Collection does not exist, creating new collection", extra={"collection_name": collection_name})
                collection = self.create_collection(
                    collection_name=collection_name,
                    name_schema=name_schema,
                    embedding_dim=embedding_dim
                )
                self._create_index(collection=collection, name_index=name_index)
            else:
                self.logger.debug("Collection already exists, loading it", extra={"collection_name": collection_name})
                collection = Collection(name=collection_name, using=self.alias)
            
            self.logger.info("Collection loaded successfully", extra={"collection_name": collection_name})
            return collection
        except Exception as e:
            self.logger.error(
                f"Error loading collection: {str(e)}",
                extra={"collection_name": collection_name, "error_type": type(e).__name__},
                exc_info=True
            )
            raise MilvusCollectionError(
                f"Error loading collection '{collection_name}': {str(e)}"
            ) from e

    def create_partition(
        self,
        *,
        collection: Collection,
        partition_name: str
    ) -> None:
        """
        Creates a partition in the collection if it doesn't exist.

        Args:
            collection: Collection where to create the partition.
            partition_name: Partition name.
        """
        try:
            logger = get_logger(__name__)
            collection_name = collection.name if hasattr(collection, 'name') else "unknown"
            
            if not collection.has_partition(partition_name=partition_name):
                collection.create_partition(partition_name=partition_name)
                logger.info(
                    "Partition created successfully",
                    extra={
                        "collection_name": collection_name,
                        "partition_name": partition_name
                    }
                )
            else:
                logger.debug(
                    "Partition already exists",
                    extra={
                        "collection_name": collection_name,
                        "partition_name": partition_name
                    }
                )
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(
                f"Error creating partition: {str(e)}",
                extra={"partition_name": partition_name, "error_type": type(e).__name__},
                exc_info=True
            )
            raise MilvusCollectionError(
                f"Error creating partition '{partition_name}': {str(e)}"
            ) from e

    def _create_index(
        self,
        *,
        collection: Collection,
        name_index: str = "default"
    ) -> None:
        """
        Creates an index to improve search efficiency.

        Args:
            collection: Collection where to create the index.
            name_index: Index name to use (default 'default').
        """
        try:
            index_params, field_name = self._indices.get_index(name_index=name_index)
            collection.create_index(
                field_name=field_name,
                index_params=index_params
            )
            self.logger.info(
                "Index created successfully",
                extra={
                    "collection_name": collection.name,
                    "name_index": name_index,
                    "field_name": field_name
                }
            )
        except IndexNotFoundError as e:
            self.logger.error(
                f"Index not found: {str(e)}",
                extra={"name_index": name_index},
                exc_info=True
            )
            raise MilvusCollectionError(str(e)) from e
        except Exception as e:
            self.logger.error(
                f"Error creating index: {str(e)}",
                extra={"name_index": name_index, "error_type": type(e).__name__},
                exc_info=True
            )
            raise MilvusCollectionError(
                f"Error creating index: {str(e)}"
            ) from e

    @staticmethod
    def release_collection(*, collection: Collection) -> None:
        """
        Releases the collection from memory.

        Args:
            collection: Collection to release.
        """
        try:
            collection.release()
        except ConnectionNotExistException:
            # Connection already closed, collection release is not needed
            # This is not an error, just skip silently
            pass
        except Exception as e:
            raise MilvusCollectionError(
                f"Error releasing collection: {str(e)}"
            ) from e
