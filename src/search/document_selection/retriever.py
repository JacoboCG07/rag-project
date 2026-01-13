"""
Module for retrieving document summaries from Milvus
"""

from typing import List, Dict, Any, Optional
from pymilvus import Collection
from src.rag.processing.milvus.milvus_client import MilvusClient
from src.utils import get_logger


class SummaryRetriever:
    """
    Class for retrieving document summaries from Milvus collection.
    Extracts necessary information from documents excluding unnecessary fields.
    """

    def __init__(
        self,
        dbname: str,
        collection_name: str,
        alias: str = "default",
        name_schema: str = "summary",
        embedding_dim: int = 1536,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None
    ):
        """
        Initializes the summary retriever.

        Args:
            dbname: Database name.
            collection_name: Summaries collection name.
            alias: Connection alias.
            name_schema: Schema name to use.
            embedding_dim: Embedding vector dimension.
            uri: Connection URI (optional).
            token: Authentication token (optional).
            host: Milvus host (optional).
            port: Milvus port (optional).
        """
        self.logger = get_logger(__name__)
        
        self.logger.info(
            "Initializing SummaryRetriever",
            extra={
                "dbname": dbname,
                "collection_name": collection_name,
                "alias": alias
            }
        )

        # Initialize Milvus client
        self.milvus_client = MilvusClient(
            dbname=dbname,
            collection_name=collection_name,
            alias=alias,
            name_schema=name_schema,
            embedding_dim=embedding_dim,
            uri=uri,
            token=token,
            host=host,
            port=port
        )
        
        # Load collection
        self.collection: Collection = self.milvus_client.load_collection()
        
        self.logger.info(
            "SummaryRetriever initialized successfully",
            extra={"collection_name": collection_name}
        )

    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """
        Gets all summaries from the collection.
        
        Excludes fields: text, file_name, type_file, total_pages, 
        total_chapters, total_num_image
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries with summary information.
            Each dictionary contains:
                - file_id: File identifier
                - file_name: File name
                - type_file: File type (PDF, etc.)
                - total_pages: Total pages
                - total_chapters: Total chapters
                - total_num_image: Total images
                - text: Summary text (description)
        """
        self.logger.info("Getting all summaries from collection")
        
        try:
            # Fields we want to retrieve (all except text_embedding and other internal fields)
            output_fields = [
                "file_id",
                "file_name", 
                "type_file",
                "total_pages",
                "total_chapters", 
                "total_num_image",
                "text"
            ]
            
            # Query all documents
            # Use high limit to get all documents
            results = self.collection.query(
                expr="id >= 0",  # Condition that includes all records
                output_fields=output_fields,
                limit=10000  # High limit to get all documents
            )
            
            self.logger.info(
                f"Retrieved {len(results)} summaries",
                extra={"count": len(results)}
            )
            
            # Process results to structure them
            summaries = []
            for result in results:
                summary = {
                    "file_id": result.get("file_id", ""),
                    "file_name": result.get("file_name", ""),
                    "type_file": result.get("type_file", ""),
                    "total_pages": result.get("total_pages", "0"),
                    "total_chapters": result.get("total_chapters", "0"),
                    "total_num_image": result.get("total_num_image", "0"),
                    "text": result.get("text", "")
                }
                summaries.append(summary)
            
            return summaries
            
        except Exception as e:
            self.logger.error(
                f"Error retrieving summaries: {str(e)}",
                extra={"error": str(e)},
                exc_info=True
            )
            raise

    def close(self) -> None:
        """Closes connection with Milvus."""
        self.logger.info("Closing Milvus connection")
        self.milvus_client.close()
        self.logger.info("Connection closed successfully")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

