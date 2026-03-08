"""
Structured logging for the Ingestion Pipeline.
"""
from pathlib import Path
from typing import Optional

from .types import ExtractionResult, BaseFileMetadata


class IngestionPipelineLogger:
    """
    Handles structured logging for the Ingestion Pipeline.
    """

    def __init__(self, logger, collection_name: str):
        self.logger = logger
        self.collection_name = collection_name

    def file_processing_start(
        self,
        file_path_obj: Path,
        file_id: str,
        extract_process_images: Optional[bool],
    ) -> None:
        """
        Logs information when starting file processing.

        Args:
            file_path_obj: Path object of the file being processed.
            file_id: File identifier.
            extract_process_images: Whether to extract images from PDFs.
        """
        self.logger.info(
            "Starting file processing",
            extra={
                "file_path": str(file_path_obj),
                "file_id": file_id,
                "collection_name": self.collection_name,
                "extract_process_images": extract_process_images,
            },
        )

    def extracted_content(
        self,
        document_data: ExtractionResult[BaseFileMetadata],
        file_id: str,
    ) -> str:
        """
        Logs information about extracted file content.

        Args:
            document_data: ExtractionResult containing the extracted document data.
            file_id: File identifier.

        Returns:
            str: File name from document metadata.
        """
        file_name = document_data.metadata.file_name
        content = len(document_data.content)
        images_count = len(document_data.images) if document_data.images else 0

        self.logger.info(
            "File content extracted",
            extra={
                "file_id": file_id,
                "file_name": file_name,
                "content": content,
                "images_count": images_count,
            },
        )
        return file_name

    def milvus_processing_start_debug(self, file_id: str, file_name: str) -> None:
        """
        Logs debug information before processing and inserting document into Milvus.

        Args:
            file_id: File identifier.
            file_name: Name of the file being processed.
        """
        self.logger.debug(
            "Processing and inserting document into Milvus",
            extra={
                "file_id": file_id,
                "file_name": file_name,
                "collection_name": self.collection_name,
            },
        )

    def file_processed_successfully(
        self,
        file_id: str,
        file_name: str,
        message: str,
    ) -> None:
        """
        Logs information when a file is processed successfully.

        Args:
            file_id: File identifier.
            file_name: Name of the file that was processed.
            message: Success message from the processing operation.
        """
        self.logger.info(
            "File processed successfully",
            extra={
                "file_id": file_id,
                "file_name": file_name,
                "collection_name": self.collection_name,
                "message": message,
            },
        )

    def file_processing_error(
        self,
        file_id: str,
        file_name: str,
        error_message: str,
    ) -> None:
        """
        Logs error information when file processing fails.

        Args:
            file_id: File identifier.
            file_name: Name of the file that failed to process.
            error_message: Error message from the processing operation.
        """
        self.logger.error(
            "Error processing file",
            extra={
                "file_id": file_id,
                "file_name": file_name,
                "collection_name": self.collection_name,
                "error_message": error_message,
            },
        )
