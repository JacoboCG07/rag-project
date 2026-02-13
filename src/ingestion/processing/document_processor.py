"""
Document processor for RAG
Orchestrates document and summary processing
"""

from src.utils import get_logger
from .milvus.milvus_client import MilvusClient
from typing import Any, Optional, Tuple, Callable
from ..types import ExtractionResult
from .uploaders import DocumentUploader, SummaryUploader


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
        port: Optional[str] = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 0,
        detect_chapters: bool = True
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
            chunk_size: Maximum size of each chunk in characters (default 2000).
            chunk_overlap: Number of characters to overlap between chunks (default 0).
            detect_chapters: Whether to detect chapters in documents (default True).
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
                "has_describe_image_func": describe_image_func is not None,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "detect_chapters": detect_chapters
            }
        )

        # Initialize DocumentUploader with embedding function, image description function, and chunking config
        self._document_uploader = DocumentUploader(
            milvus_client=self.milvus_client,
            generate_embeddings_func=generate_embeddings_func,
            describe_image_func=describe_image_func,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            detect_chapters=detect_chapters
        )
        
        # Initialize SummaryProcessor (generate_summary_func is always provided via config)
        self._summary_uploader = SummaryUploader(
            milvus_client=self.milvus_client,
            generate_summary_func=generate_summary_func,
            generate_embeddings_func=generate_embeddings_func
        )

        self.logger.debug("SummaryProcessor initialized")
        self.logger.debug("DocumentProcessor initialized successfully")


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
            file_name = document_data.metadata.file_name

            # Upload document
            self._log_document_processing_start(file_id, file_name, process_images)
            success_doc, message_doc = self._document_uploader.upload_document(
                document_data=document_data,
                file_id=file_id,
                process_images=process_images,
                partition_name=self.PARTITION_DOCUMENTS
            )

            if not success_doc:
                self._log_document_upload_error(file_id, file_name, message_doc)
                return False, message_doc

            self._log_document_upload_success(file_id, file_name, message_doc)

            # Upload summary
            self.logger.debug("Processing summary", extra={"file_id": file_id})
            success_summary, message_summary = self._summary_uploader.upload_summary(
                document_data=document_data,
                file_id=file_id,
                partition_name=self.PARTITION_SUMMARIES
            )

            if not success_summary:
                self._log_summary_upload_error(file_id, file_name, message_summary)
                return False, f"{message_doc}, but summary failed: {message_summary}"

            self._log_document_and_summary_success(file_id, file_name, message_doc, message_summary)

            return True, f"{message_doc}, {message_summary}"


        except Exception as e:
            error_msg = self._log_and_get_processing_error(file_id, document_data, e)
            return False, error_msg


    def _log_document_processing_start(
        self,
        file_id: str,
        file_name: str,
        process_images: bool
    ) -> None:
        """
        Logs information when starting document processing and insertion.

        Args:
            file_id: File identifier.
            file_name: Name of the file being processed.
            process_images: Whether to process images.
        """
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

    def _log_document_upload_error(
        self,
        file_id: str,
        file_name: str,
        error_message: str
    ) -> None:
        """
        Logs error information when document upload fails.

        Args:
            file_id: File identifier.
            file_name: Name of the file that failed to upload.
            error_message: Error message from the upload operation.
        """
        self.logger.error(
            "Document upload failed",
            extra={
                "file_id": file_id,
                "file_name": file_name,
                "error_message": error_message
            }
        )

    def _log_document_upload_success(
        self,
        file_id: str,
        file_name: str,
        message: str
    ) -> None:
        """
        Logs information when document upload is successful.

        Args:
            file_id: File identifier.
            file_name: Name of the file that was uploaded.
            message: Success message from the upload operation.
        """
        self.logger.info(
            "Document uploaded successfully",
            extra={
                "file_id": file_id,
                "file_name": file_name,
                "message": message
            }
        )

    def _log_summary_upload_error(
        self,
        file_id: str,
        file_name: str,
        error_message: str
    ) -> None:
        """
        Logs error information when summary processing fails.

        Args:
            file_id: File identifier.
            file_name: Name of the file that failed to process summary.
            error_message: Error message from the summary processing operation.
        """
        self.logger.error(
            "Summary processing failed",
            extra={
                "file_id": file_id,
                "file_name": file_name,
                "error_message": error_message
            }
        )

    def _log_document_and_summary_success(
        self,
        file_id: str,
        file_name: str,
        document_message: str,
        summary_message: str
    ) -> None:
        """
        Logs information when document and summary are processed successfully.

        Args:
            file_id: File identifier.
            file_name: Name of the file that was processed.
            document_message: Success message from the document upload operation.
            summary_message: Success message from the summary processing operation.
        """
        self.logger.info(
            "Document and summary processed successfully",
            extra={
                "file_id": file_id,
                "file_name": file_name,
                "document_message": document_message,
                "summary_message": summary_message
            }
        )

    def _log_and_get_processing_error(
        self,
        file_id: str,
        document_data: ExtractionResult,
        exception: Exception
    ) -> str:
        """
        Logs error information when document processing fails and returns error message.

        Args:
            file_id: File identifier.
            document_data: ExtractionResult containing document data.
            exception: Exception that was raised during processing.

        Returns:
            str: Error message to return.
        """
        # Get file_name from metadata if available for error message
        try:
            file_name = document_data.metadata.file_name
        except Exception:
            file_name = "unknown"

        error_msg = f"Error processing document {file_name}: {str(exception)}"
        self.logger.error(
            error_msg,
            extra={
                "file_id": file_id,
                "file_name": file_name,
                "error_type": type(exception).__name__
            },
            exc_info=True
        )
        return error_msg

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


