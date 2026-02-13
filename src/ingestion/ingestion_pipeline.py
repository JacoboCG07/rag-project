"""
Ingestion Pipeline
Integrates DocumentExtractionManager with DocumentProcessor for end-to-end document processing
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import hashlib

from .types import ExtractionResult, BaseFileMetadata
from .extractors import DocumentExtractionManager
from .processing.document_processor import DocumentProcessor
from .config import IngestionPipelineConfig
from src.utils import get_logger


class IngestionPipeline:
    """
    High-level pipeline that integrates DocumentExtractionManager with DocumentProcessor.
    Provides end-to-end document processing from folder to Milvus.
    """

    def __init__(
        self,
        *,
        config: IngestionPipelineConfig
    ):
        """
        Initializes the Ingestion Pipeline.

        Args:
            config: IngestionPipelineConfig with all configuration parameters.
            generate_embeddings_func: Function to generate embeddings (must receive text and return embedding).
            generate_summary_func: Function to generate summaries (optional, required if generate_summary=True in config).
        """
        self.logger = get_logger(__name__)
        self.config = config
        
        self.logger.info(
            "Initializing Ingestion Pipeline",
            extra={
                "milvus_db": config.milvus.dbname,
                "collection_name": config.collection_name,
                "embedding_dim": config.embedder.dimensions,
                "milvus_host": config.milvus.host or "default",
                "milvus_port": config.milvus.port or "default",
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "extract_images": config.extract_images
            }
        )
        
        # Initialize DocumentProcessor with config
        try:
            self.document_processor = DocumentProcessor(
                dbname=config.milvus.dbname,
                collection_name=config.collection_name,
                generate_embeddings_func=config.generate_embeddings_func,
                generate_summary_func=config.generate_summary_func,
                describe_image_func=config.describe_image_func,
                alias=config.milvus.alias,
                embedding_dim=config.embedder.dimensions,
                uri=config.milvus.uri,
                token=config.milvus.token,
                host=config.milvus.host,
                port=config.milvus.port,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                detect_chapters=config.detect_chapters
            )
            self.logger.info("Ingestion Pipeline initialized successfully")
        except Exception as e:
            self.logger.error(
                f"Error initializing Ingestion Pipeline: {str(e)}",
                extra={
                    "milvus_db": config.milvus.dbname,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            raise
        
    def process_single_file(
        self,
        *,
        file_path: str,
        extract_process_images: Optional[bool] = None,
        job_id: Optional[str] = None

    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Processes a single file and inserts it into Milvus.
        Documents are inserted into the 'documents' partition and summaries into 'summaries'
        of the collection specified in the configuration.

        Args:
            file_path: Path to the file to process.
            extract_process_images: Whether to extract images from PDFs (overrides config default).
            job_id: Optional job identifier for tracking logs. If provided, will be included in all logs.

        Returns:
            Tuple[bool, str, Dict]: (success, message, result_info).
            result_info contains: file_id, file_name, file_path.
        """
        # Bind job_id al logger para que aparezca en todos los logs
        if job_id: self.logger = self.logger.bind(job_id=job_id)
        
        file_path_obj = Path(file_path)
        file_id = self._generate_file_id(file_path)
        self._log_file_processing_start(file_path_obj, file_id, extract_process_images)
        
        try:
            # Initialize and extract content from DocumentExtractionManager with parent folder
            extraction_manager = DocumentExtractionManager(folder_path=str(file_path_obj.parent))
            document_data: ExtractionResult[BaseFileMetadata] = extraction_manager.extract_file_data(
                file_path=file_path_obj,
                extract_images=extract_process_images
            )
            file_name = self._log_extracted_content(document_data, file_id)

            # Process and insert document in milvus
            self._log_milvus_processing_start_debug(file_id, file_name)
            success, message = self.document_processor.process_and_insert(
                file_id=file_id,
                document_data=document_data,
                process_images=extract_process_images
            )
            
            # Prepare result info
            result_info = {
                "file_id": file_id,
                "file_name": document_data.metadata.file_name,
                "file_path": str(file_path_obj)
            }
            
            if success: self._log_file_processed_successfully(file_id, file_name, message)
            else: self._log_file_processing_error(file_id, file_name, message)
            
            return success, message, result_info
            
        except Exception as e:
            self.logger.error(
                f"Exception processing file: {str(e)}",
                extra={
                    "file_path": str(file_path_obj),
                    "file_id": file_id,
                    "collection_name": self.config.collection_name,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            # Prepare result info even on error
            result_info = {
                "file_id": file_id,
                "file_name": file_path_obj.name,
                "file_path": str(file_path_obj)
            }
            return False, f"Error processing file: {str(e)}", result_info


    @staticmethod
    def _generate_file_id(file_path: str) -> str:
        """
        Generates a consistent file_id from file path using hash.

        Args:
            file_path: Path to the file.

        Returns:
            str: File ID (hash-based).
        """
        # Normalize path and generate hash
        normalized_path = str(Path(file_path).resolve())
        file_hash = hashlib.md5(normalized_path.encode()).hexdigest()
        return file_hash

    def _log_file_processing_start(
        self,
        file_path_obj: Path,
        file_id: str,
        extract_process_images: Optional[bool]
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
                "collection_name": self.config.collection_name,
                "extract_process_images": extract_process_images
            }
        )

    def _log_extracted_content(
        self,
        document_data: ExtractionResult[BaseFileMetadata],
        file_id: str
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
                "images_count": images_count
            }
        )
        return file_name

    def _log_milvus_processing_start_debug(
        self,
        file_id: str,
        file_name: str
    ) -> None:
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
                "collection_name": self.config.collection_name
            }
        )

    def _log_file_processed_successfully(
        self,
        file_id: str,
        file_name: str,
        message: str
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
                "collection_name": self.config.collection_name,
                "message": message
            }
        )

    def _log_file_processing_error(
        self,
        file_id: str,
        file_name: str,
        error_message: str
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
                "collection_name": self.config.collection_name,
                "error_message": error_message
            }
        )

    def close(self) -> None:
        """Closes connections with Milvus."""
        self.logger.info("Closing Ingestion Pipeline connections")
        try:
            self.document_processor.close()
            self.logger.info("Ingestion Pipeline connections closed successfully")
        except Exception as e:
            self.logger.error(
                f"Error closing RAG Pipeline connections: {str(e)}",
                extra={"error_type": type(e).__name__},
                exc_info=True
            )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.logger.error(
                f"Exception in context manager: {exc_type.__name__}",
                extra={
                    "exception_type": exc_type.__name__,
                    "exception_value": str(exc_val) if exc_val else None
                },
                exc_info=True
            )
        self.close()