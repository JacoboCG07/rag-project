"""
Document uploader for Milvus
Orchestrates text and image processing and uploading
"""

from typing import Tuple, Callable, Any, Optional
from src.utils import get_logger
from ..milvus.milvus_client import MilvusClient
from ...types import ExtractionResult
from .components import TextUploader, ImageUploader


class DocumentUploader:
    """
    Orchestrates processing and uploading of full documents to Milvus.
    Delegates text and image processing to specialized components.
    """

    def __init__(
        self,
        *,
        milvus_client: MilvusClient,
        generate_embeddings_func: Callable[[str], Any],
        describe_image_func: Optional[Callable[[str], str]] = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 0,
        detect_chapters: bool = True,
    ):
        """
        Initializes the document uploader.

        Args:
            milvus_client: Milvus client for documents collection.
            generate_embeddings_func: Function to generate embeddings.
            describe_image_func: Function to describe image (optional).
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
            detect_chapters: Whether to detect chapters in documents.
        """
        self.logger = get_logger(__name__)

        # Initialize specialized uploaders
        self._text_uploader = TextUploader(
            milvus_client=milvus_client,
            generate_embeddings_func=generate_embeddings_func,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            detect_chapters=detect_chapters,
        )

        self._image_uploader = ImageUploader(
            milvus_client=milvus_client,
            generate_embeddings_func=generate_embeddings_func,
            describe_image_func=describe_image_func,
        )

        self.logger.info(
            "Initializing DocumentUploader",
            extra={
                "has_describe_image_func": describe_image_func is not None,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "detect_chapters": detect_chapters,
            },
        )

    def upload_document(
        self,
        *,
        document_data: ExtractionResult,
        file_id: str,
        process_images: bool = False,
        partition_name: str,
    ) -> Tuple[bool, str]:
        """
        Processes and uploads a document to Milvus.

        Args:
            document_data: ExtractionResult with 'content' (list of texts), 
                          'images' (optional list of ImageData), 
                          and 'metadata' (BaseFileMetadata or subclass).
            file_id: Unique file ID.
            process_images: Whether to process and vectorize images.
            partition_name: Partition name for Milvus.

        Returns:
            Tuple[bool, str]: (success, message).

        Raises:
            ValueError: If document_data doesn't have the expected format.
        """
        try:
            self._log_starting_document_upload(
                file_id=file_id,
                process_images=process_images,
                partition_name=partition_name,
            )

            # Extract data from ExtractionResult
            content = document_data.content
            images = document_data.images or []
            metadata_obj = document_data.metadata
            file_name = metadata_obj.file_name
            file_type = (
                metadata_obj.file_type
                if hasattr(metadata_obj, "file_type")
                else "document"
            )

            if not content:
                self.logger.error(
                    "document_data must contain 'content' with at least one element"
                )
                raise ValueError(
                    "document_data must contain 'content' with at least one element"
                )

            self._log_processing_document(
                file_id=file_id,
                file_name=file_name,
                content_chunks=len(content),
                images_count=len(images),
            )

            # Upload texts using TextUploader
            chunks, chunks_count = self._text_uploader.process_and_upload(
                content=content,
                file_id=file_id,
                file_name=file_name,
                file_type=file_type,
                partition_name=partition_name,
            )

            # Upload images if requested using ImageUploader
            processed_images_count = 0
            if process_images and images:
                processed_images_count = self._image_uploader.process_and_upload(
                    images=images,
                    file_id=file_id,
                    file_name=file_name,
                    file_type=file_type,
                    partition_name=partition_name,
                )

            # Build success message
            message = f"Document {file_name} uploaded successfully"
            if process_images and images and processed_images_count > 0:
                message += f" (with {processed_images_count} images processed)"

            self.logger.info(
                "Document uploaded successfully",
                extra={
                    "file_id": file_id,
                    "file_name": file_name,
                    "chunks_count": chunks_count,
                    "images_processed": processed_images_count,
                    "partition_name": partition_name,
                },
            )

            return True, message

        except Exception as e:
            # Get file_name for error message (in case it wasn't extracted yet)
            try:
                file_name = (
                    document_data.metadata.file_name
                    if hasattr(document_data, "metadata")
                    else "unknown"
                )
            except:
                file_name = "unknown"

            error_msg = f"Error uploading document {file_name}: {str(e)}"
            self.logger.error(
                error_msg,
                extra={
                    "file_id": file_id,
                    "file_name": file_name,
                    "partition_name": partition_name,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            return False, error_msg

    def _log_starting_document_upload(
        self,
        *,
        file_id: str,
        process_images: bool,
        partition_name: str,
    ) -> None:
        """
        Logs the start of a document upload process.

        Args:
            file_id: Unique identifier of the file.
            process_images: Whether images will be processed.
            partition_name: Target partition in Milvus.
        """
        self.logger.info(
            "Starting document upload",
            extra={
                "file_id": file_id,
                "process_images": process_images,
                "partition_name": partition_name,
            },
        )

    def _log_processing_document(
        self,
        *,
        file_id: str,
        file_name: str,
        content_chunks: int,
        images_count: int,
    ) -> None:
        """
        Logs information about the document being processed.

        Args:
            file_id: Unique identifier of the file.
            file_name: Name of the file.
            content_chunks: Number of content chunks/pages.
            images_count: Number of images associated with the document.
        """
        self.logger.debug(
            "Processing document",
            extra={
                "file_id": file_id,
                "file_name": file_name,
                "content_chunks": content_chunks,
                "images_count": images_count,
            },
        )
