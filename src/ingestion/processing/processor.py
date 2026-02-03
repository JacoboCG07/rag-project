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
        collection_name: str,
        generate_embeddings_func: Callable[[str], Any],
        generate_summary_func: Callable[[str], str],
        describe_image_func: Callable[[str], str] = None,
        alias: str = "default",
        embedding_dim: int = 1536,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None
    ):
        """
        Initializes the document processor with Milvus client.
        Usa una sola colección con dos particiones fijas: 'documents' y 'summaries'.

        Args:
            dbname: Database name in Milvus.
            collection_name: Nombre de la colección. Tendrá dos particiones: 'documents' y 'summaries'.
            generate_embeddings_func: Function to generate embeddings (must receive text and return embedding).
            generate_summary_func: Function to generate summary (must receive full text and return summary string). Required.
            describe_image_func: Function to describe image (must receive base64 image and return description string). Optional.
            alias: Connection alias.
            embedding_dim: Embedding vector dimension.
            uri: Connection URI (optional).
            token: Authentication token (optional).
            host: Milvus host (optional).
            port: Milvus port (optional).
        """

        # Usar una sola colección para documentos y resúmenes
        # Las particiones fijas serán 'documents' y 'summaries'
        self.milvus_client = MilvusClient(
            dbname=dbname,
            collection_name=collection_name,
            alias=alias,
            name_schema="document",  # Usamos el schema de documento para ambos
            embedding_dim=embedding_dim,
            uri=uri,
            token=token,
            host=host,
            port=port
        )
        
        # Particiones fijas
        self.PARTITION_DOCUMENTS = "documents"
        self.PARTITION_SUMMARIES = "summaries"

        # Store embedding and summary generation functions
        self.generate_embeddings_func = generate_embeddings_func
        self.generate_summary_func = generate_summary_func
        self.describe_image_func = describe_image_func
        self.logger = get_logger(__name__)

        self.logger.info(
            "Initializing DocumentProcessor",
            extra={
                "dbname": dbname,
                "collection_name": collection_name,
                "partition_documents": self.PARTITION_DOCUMENTS,
                "partition_summaries": self.PARTITION_SUMMARIES,
                "embedding_dim": embedding_dim,
                "has_describe_image_func": describe_image_func is not None
            }
        )

        # Initialize DocumentUploader with embedding function and image description function
        self._document_uploader = DocumentUploader(
            milvus_client=self.milvus_client,
            generate_embeddings_func=generate_embeddings_func,
            describe_image_func=describe_image_func
        )
        
        # Initialize SummaryProcessor (generate_summary_func is always provided via config)
        self._summary_processor = SummaryProcessor(
            milvus_client=self.milvus_client,
            generate_summary_func=generate_summary_func,
            generate_embeddings_func=generate_embeddings_func
        )
        self.logger.debug("SummaryProcessor initialized")
        
        self.logger.info("DocumentProcessor initialized successfully")

    def process_and_insert(
        self,
        *,
        file_id: str,
        document_data: ExtractionResult,
        process_images: bool = False
    ) -> Tuple[bool, str]:
        """
        Processes document data and inserts it into Milvus collections.
        Always inserts full documents and generates summaries. Optionally processes images.
        Usa las particiones fijas 'documents' y 'summaries'.

        Args:
            file_id: Unique file ID.
            document_data: ExtractionResult with 'content' (list of texts), 'images' (optional list of ImageData),
                          and 'metadata' (BaseFileMetadata or subclass).
            process_images: Whether to process and vectorize images (default False).

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
                    "partition_documents": self.PARTITION_DOCUMENTS,
                    "partition_summaries": self.PARTITION_SUMMARIES
                }
            )

            success_doc, message_doc = self._document_uploader.upload_document(
                document_data=document_data,
                file_id=file_id,
                process_images=process_images,
                partition_name=self.PARTITION_DOCUMENTS
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

            # Process summary (always available)
            self.logger.debug("Processing summary", extra={"file_id": file_id})
            success_summary, message_summary = self._summary_processor.process_and_upload_summary(
                document_data=document_data,
                file_id=file_id,
                partition_name=self.PARTITION_SUMMARIES
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
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            return False, error_msg

    def close(self) -> None:
        """Closes connections with Milvus collection."""
        self.logger.info("Closing DocumentProcessor connections")
        try:
            self.milvus_client.close()
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