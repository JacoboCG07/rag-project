"""
Module for retrieving document summaries from Milvus
"""

from typing import List, Optional
from pymilvus import Collection
from src.rag.processing.milvus.milvus_client import MilvusClient
from src.search.models import DocumentSummary
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

    def get_all_summaries(self) -> List[DocumentSummary]:
        """
        Gets all summaries from the collection.
        
        Returns:
            List[DocumentSummary]: List of document summaries with metadata.
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
            
            # Query all documents from the 'summaries' partition
            # Use high limit to get all documents
            results = self.collection.query(
                expr="id >= 0",  # Condition that includes all records
                output_fields=output_fields,
                partition_names=["summaries"],  # Buscar solo en la partición de resúmenes
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

    def get_summaries_by_file_ids(self, file_ids: List[str]) -> List[DocumentSummary]:
        """
        Gets summaries from the collection filtered by specific file_ids.
        
        Uses Milvus query with filter expression to retrieve only the requested documents.
        
        Args:
            file_ids: List of file_id strings to retrieve.
        
        Returns:
            List[DocumentSummary]: List of document summaries for the requested IDs.
        """
        if not file_ids:
            self.logger.warning("No file_ids provided, returning empty list")
            return []
        
        self.logger.info(
            "Getting summaries by file_ids",
            extra={"file_ids_count": len(file_ids), "file_ids": file_ids}
        )
        
        try:
            # Fields we want to retrieve
            output_fields = [
                "file_id",
                "file_name", 
                "type_file",
                "total_pages",
                "total_chapters", 
                "total_num_image",
                "text"
            ]
            
            # Build filter expression for file_ids
            if len(file_ids) == 1:
                filter_expr = f'file_id == "{file_ids[0]}"'
            else:
                # Build: file_id in ["id1", "id2", ...]
                file_ids_str = ", ".join([f'"{fid}"' for fid in file_ids])
                filter_expr = f'file_id in [{file_ids_str}]'
            
            self.logger.debug(f"Querying Milvus with filter: {filter_expr}")
            
            # Query with filter from the 'summaries' partition
            results = self.collection.query(
                expr=filter_expr,
                output_fields=output_fields,
                partition_names=["summaries"],  # Buscar solo en la partición de resúmenes
                limit=10000  # High limit to get all matching documents
            )
            
            self.logger.info(
                f"Retrieved {len(results)} summaries for {len(file_ids)} file_ids",
                extra={"count": len(results), "requested_count": len(file_ids)}
            )
            
            # Process results to structure them
            summaries = []
            for result in results:
                summaries.append(DocumentSummary(
                    file_id=result.get("file_id", ""),
                    file_name=result.get("file_name", ""),
                    type_file=result.get("type_file", ""),
                    total_pages=result.get("total_pages", "0"),
                    total_chapters=result.get("total_chapters", "0"),
                    total_num_image=result.get("total_num_image", "0"),
                    text=result.get("text", "")
                ))
            
            return summaries
            
        except Exception as e:
            self.logger.error(
                f"Error retrieving summaries by file_ids: {str(e)}",
                extra={"error": str(e), "file_ids": file_ids},
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

