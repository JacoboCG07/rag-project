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
            schema = self._schemas.get_schema(name_schema=name_schema, embedding_dim=embedding_dim)
            collection = Collection(
                name=collection_name,
                schema=schema,
                using=self.alias
            )
            return collection
        except SchemaNotFoundError as e:
            raise MilvusCollectionError(str(e)) from e
        except Exception as e:
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
            collections = utility.list_collections(using=self.alias)
            
            if collection_name not in collections:
                collection = self.create_collection(
                    collection_name=collection_name,
                    name_schema=name_schema,
                    embedding_dim=embedding_dim
                )
                self._create_index(collection=collection, name_index=name_index)
            else:
                collection = Collection(name=collection_name, using=self.alias)
            
            return collection
        except Exception as e:
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
            if not collection.has_partition(partition_name=partition_name):
                collection.create_partition(partition_name=partition_name)
        except Exception as e:
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
        except IndexNotFoundError as e:
            raise MilvusCollectionError(str(e)) from e
        except Exception as e:
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
