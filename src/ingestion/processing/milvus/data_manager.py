"""
Milvus data manager
Single responsibility: data preparation and insertion
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pymilvus import Collection

from .exceptions import MilvusInsertError
from src.utils import get_logger


class DataManager:
    """
    Manages data insertion into Milvus.
    Single responsibility: data preparation and insertion.
    """

    @staticmethod
    def prepare_data_for_insertion(
        *,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepares data for insertion into Milvus.

        Args:
            texts: List of texts to insert.
            embeddings: List of corresponding embeddings.
            metadata: Additional metadata (optional).

        Returns:
            List[Dict]: List of dictionaries prepared for insertion.

        Raises:
            MilvusInsertError: If there's an error in preparation.
        """
        logger = get_logger(__name__)
        
        if len(texts) != len(embeddings):
            error_msg = "Number of texts must match number of embeddings"
            logger.error(error_msg, extra={"texts_count": len(texts), "embeddings_count": len(embeddings)})
            raise MilvusInsertError(error_msg)

        try:
            logger.debug(
                "Preparing data for insertion",
                extra={
                    "texts_count": len(texts),
                    "has_metadata": bool(metadata)
                }
            )
            data_list = []
            current_date = datetime.now().date().strftime('%Y-%m-%d')
            metadata = metadata or {}

            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                data = {
                    "text": text,
                    "text_embedding": embedding,
                }

                # Add metadata
                for key, value in metadata.items():
                    if isinstance(value, list) and i < len(value):
                        # If it's a list, take the value corresponding to the index
                        data[key] = str(value[i]) if not isinstance(value[i], str) else value[i]
                    elif isinstance(value, str):
                        data[key] = value
                    else:
                        data[key] = str(value) if value is not None else ""

                data_list.append(data)

            logger.debug("Data prepared successfully", extra={"data_count": len(data_list)})
            return data_list
        except Exception as e:
            logger.error(
                f"Error preparing data for insertion: {str(e)}",
                extra={"texts_count": len(texts), "error_type": type(e).__name__},
                exc_info=True
            )
            raise MilvusInsertError(
                f"Error preparing data for insertion: {str(e)}"
            ) from e

    @staticmethod
    def insert_data(
        *,
        collection: Collection,
        data: List[Dict[str, Any]],
        partition_name: Optional[str] = None
    ) -> None:
        """
        Inserts data into the Milvus collection.

        Args:
            collection: Collection where to insert.
            data: Data to insert.
            partition_name: Partition name (optional).

        Raises:
            MilvusInsertError: If there's an error in insertion.
        """
        logger = get_logger(__name__)
        collection_name = collection.name if hasattr(collection, 'name') else "unknown"
        
        if not data:
            error_msg = "No data to insert"
            logger.error(error_msg)
            raise MilvusInsertError(error_msg)

        try:
            if partition_name:
                collection.insert(data=data, partition_name=partition_name)
            else:
                collection.insert(data=data)
            
            logger.info(
                "Data inserted successfully",
                extra={
                    "collection_name": collection_name,
                    "data_count": len(data),
                    "partition_name": partition_name
                }
            )
        except Exception as e:
            logger.error(
                f"Error inserting data: {str(e)}",
                extra={
                    "collection_name": collection_name,
                    "data_count": len(data),
                    "partition_name": partition_name,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            raise MilvusInsertError(
                f"Error inserting data into partition '{partition_name}': {str(e)}"
            ) from e
