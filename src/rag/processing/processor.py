"""
Document processor for RAG
Orchestrates document and summary processing
"""

from .milvus.milvus_client import MilvusClient
from typing import Any, Optional, Tuple, Callable
from .uploaders import DocumentUploader, SummaryProcessor
from ..extractors.base.types import ExtractionResult, BaseFileMetadata
from src.utils import get_logger

class DocumentProcessor:
    """
    Orchestrator for document processing and insertion into Milvus.
    Coordinates DocumentUploader and SummaryProcessor.
    """

    def __init__(
        self,
        *,
        dbname: str,
        collection_name_documents: str,
        collection_name_summaries: str,
        generate_embeddings_func: Callable[[str], Any],
        generate_summary_func: Callable[[str], str] = None,
        describe_image_func: Callable[[str], str] = None,
        alias: str = "default",
        embedding_dim: int = 1536,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None
    ):
        """
        Initializes the document processor with Milvus clients.

        Args:
            dbname: Database name in Milvus.
            collection_name_documents: Collection name for full documents.
            collection_name_summaries: Collection name for document summaries.
            generate_embeddings_func: Function to generate embeddings (must receive text and return embedding).
            generate_summary_func: Function to generate summary (must receive full text and return summary string). Optional.
            describe_image_func: Function to describe image (must receive base64 image and return description string). Optional.
            alias: Connection alias.
            embedding_dim: Embedding vector dimension.
            uri: Connection URI (optional).
            token: Authentication token (optional).
            host: Milvus host (optional).
            port: Milvus port (optional).
        """

        self.milvus_client_documents = MilvusClient(
            dbname=dbname,
            collection_name=collection_name_documents,
            alias=alias,
            name_schema="document",
            embedding_dim=embedding_dim,
            uri=uri,
            token=token,
            host=host,
            port=port
        )
        
        self.milvus_client_summaries = MilvusClient(
            dbname=dbname,
            collection_name=collection_name_summaries,
            alias=alias,
            name_schema="summary",
            embedding_dim=embedding_dim,
            uri=uri,
            token=token,
            host=host,
            port=port
        )

        # Store embedding and summary generation functions
        self.generate_embeddings_func = generate_embeddings_func
        self.generate_summary_func = generate_summary_func
        self.describe_image_func = describe_image_func
        self.logger = get_logger(__name__)

        self.logger.info(
            "Initializing DocumentProcessor",
            extra={
                "dbname": dbname,
                "collection_name_documents": collection_name_documents,
                "collection_name_summaries": collection_name_summaries,
                "embedding_dim": embedding_dim,
                "has_generate_summary_func": generate_summary_func is not None,
                "has_describe_image_func": describe_image_func is not None
            }
        )

        # Initialize DocumentUploader with embedding function and image description function
        self._document_uploader = DocumentUploader(
            milvus_client=self.milvus_client_documents,
            generate_embeddings_func=generate_embeddings_func,
            describe_image_func=describe_image_func
        )
        
        # Initialize SummaryProcessor only if generate_summary_func is provided
        if generate_summary_func:
            self._summary_processor = SummaryProcessor(
                milvus_client=self.milvus_client_summaries,
                generate_summary_func=generate_summary_func,
                generate_embeddings_func=generate_embeddings_func
            )
            self.logger.debug("SummaryProcessor initialized")
        else:
            self._summary_processor = None
            self.logger.debug("SummaryProcessor not initialized (no generate_summary_func provided)")
        
        self.logger.info("DocumentProcessor initialized successfully")

    def process_and_insert(
        self,
        *,
        file_id: str,
        document_data: ExtractionResult,
        process_images: bool = False,
        partition_name: str
    ) -> Tuple[bool, str]:
        """
        Processes document data and inserts it into Milvus collections.
        Always inserts full documents, optionally processes images and generates summaries.

        Args:
            file_id: Unique file ID.
            document_data: ExtractionResult with 'content' (list of texts), 'images' (optional list of ImageData),
                          and 'metadata' (BaseFileMetadata or subclass).
            process_images: Whether to process and vectorize images (default False).
            partition_name: Partition name for Milvus.

        Returns:
            Tuple[bool, str]: (success, message).

        Raises:
            ValueError: If document_data doesn't have the expected format or parameters are invalid.
        """
        try:
            # Get file_name from metadata for error messages
            file_name = document_data.metadata.file_name if hasattr(document_data, 'metadata') else "unknown"
            
            self.logger.info(
                "Starting document processing and insertion",
                extra={
                    "file_id": file_id,
                    "file_name": file_name,
                    "process_images": process_images,
                    "partition_name": partition_name,
                    "has_summary_processor": self._summary_processor is not None
                }
            )

            success_doc, message_doc = self._document_uploader.upload_document(
                document_data=document_data,
                file_id=file_id,
                process_images=process_images,
                partition_name=partition_name
            )

            if not success_doc:
                self.logger.error(
                    "Document upload failed",
                    extra={
                        "file_id": file_id,
                        "file_name": file_name,
                        "error_message": message_doc
                    }
                )
                return False, message_doc

            self.logger.info(
                "Document uploaded successfully",
                extra={
                    "file_id": file_id,
                    "file_name": file_name,
                    "message": message_doc
                }
            )

            if self._summary_processor:
                self.logger.debug("Processing summary", extra={"file_id": file_id})
                success_summary, message_summary = self._summary_processor.process_and_upload_summary(
                    document_data=document_data,
                    file_id=file_id,
                    partition_name=partition_name
                )

                if not success_summary:
                    self.logger.error(
                        "Summary processing failed",
                        extra={
                            "file_id": file_id,
                            "file_name": file_name,
                            "error_message": message_summary
                        }
                    )
                    return False, f"{message_doc}, but summary failed: {message_summary}"

                self.logger.info(
                    "Document and summary processed successfully",
                    extra={
                        "file_id": file_id,
                        "file_name": file_name,
                        "document_message": message_doc,
                        "summary_message": message_summary
                    }
                )
                return True, f"{message_doc}, {message_summary}"
            else:
                self.logger.info(
                    "Document processed successfully (no summary)",
                    extra={
                        "file_id": file_id,
                        "file_name": file_name,
                        "message": message_doc
                    }
                )
                return True, message_doc


        except Exception as e:
            # Get file_name from metadata if available for error message
            try:
                file_name = document_data.metadata.file_name if hasattr(document_data, 'metadata') else "unknown"
            except:
                file_name = "unknown"
            error_msg = f"Error processing document {file_name}: {str(e)}"
            self.logger.error(
                error_msg,
                extra={
                    "file_id": file_id,
                    "file_name": file_name,
                    "partition_name": partition_name,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            return False, error_msg

    def close(self) -> None:
        """Closes connections with both Milvus collections."""
        self.logger.info("Closing DocumentProcessor connections")
        try:
            self.milvus_client_documents.close()
            self.milvus_client_summaries.close()
            self.logger.info("DocumentProcessor connections closed successfully")
        except Exception as e:
            self.logger.error(
                f"Error closing DocumentProcessor connections: {str(e)}",
                extra={"error_type": type(e).__name__},
                exc_info=True
            )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()