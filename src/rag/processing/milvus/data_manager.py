"""
Milvus data manager
Single responsibility: data preparation and insertion
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pymilvus import Collection

from .exceptions import MilvusInsertError


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
        tokens: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepares data for insertion into Milvus.

        Args:
            texts: List of texts to insert.
            embeddings: List of corresponding embeddings.
            tokens: List of tokens (optional).
            metadata: Additional metadata (optional).

        Returns:
            List[Dict]: List of dictionaries prepared for insertion.

        Raises:
            MilvusInsertError: If there's an error in preparation.
        """
        if len(texts) != len(embeddings):
            raise MilvusInsertError(
                "Number of texts must match number of embeddings"
            )

        try:
            data_list = []
            current_date = datetime.now().date().strftime('%Y-%m-%d')
            metadata = metadata or {}

            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                data = {
                    "text": text,
                    "text_embedding": embedding,
                    "tokens": tokens[i] if tokens and i < len(tokens) else "",
                    "image_embedding": "",
                    "audio_embedding": "",
                    "date": current_date,
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

            return data_list
        except Exception as e:
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
        if not data:
            raise MilvusInsertError("No data to insert")

        try:
            if partition_name:
                collection.insert(data=data, partition_name=partition_name)
            else:
                collection.insert(data=data)
        except Exception as e:
            raise MilvusInsertError(
                f"Error inserting data into partition '{partition_name}': {str(e)}"
            ) from e
