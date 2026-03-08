"""
Ingestion Pipeline
Integrates DocumentExtractionManager with DocumentProcessor for end-to-end document processing
"""

import hashlib
from pathlib import Path
from .config import IngestionPipelineConfig
from src.utils import get_logger, set_job_id
from typing import Dict, Any, Optional, Tuple
from .extractors import DocumentExtractionManager
from .pipeline_logger import IngestionPipelineLogger
from .types import ExtractionResult, BaseFileMetadata
from .processing.document_processor import DocumentProcessor

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
        self.pipeline_logger = IngestionPipelineLogger(self.logger, config.collection_name)
        
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
                "extract_images": config.extract_images,
                "max_images_per_document": config.max_images_per_document
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
                chunk_overlap=config.chunk_overlap
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
        # Propagate job_id: bind to logger and contextvar (so all subcomponent logs include it in MongoDB)
        if job_id:
            self.logger = self.logger.bind(job_id=job_id)
            self.pipeline_logger.logger = self.logger
            set_job_id(job_id)

        # Use config.extract_images as fallback when not explicitly specified
        if extract_process_images is None:
            extract_process_images = self.config.extract_images

        file_path_obj = Path(file_path)
        file_id = self._generate_file_id(file_path)
        self.pipeline_logger.file_processing_start(file_path_obj, file_id, extract_process_images)

        try:
            # Initialize and extract content from DocumentExtractionManager with parent folder
            extraction_manager = DocumentExtractionManager(folder_path=str(file_path_obj.parent))
            document_data: ExtractionResult[BaseFileMetadata] = extraction_manager.extract_file_data(
                file_path=file_path_obj,
                extract_images=extract_process_images
            )
            file_name = self.pipeline_logger.extracted_content(document_data, file_id)

            limit_exceeded, error_message, result_info = self._check_max_images_limit(
                extract_process_images, document_data, file_id, file_name, file_path_obj
            )
            if limit_exceeded: return False, error_message, result_info

            # Process and insert document in milvus
            self.pipeline_logger.milvus_processing_start_debug(file_id, file_name)
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
            
            if success: self.pipeline_logger.file_processed_successfully(file_id, file_name, message)
            else: self.pipeline_logger.file_processing_error(file_id, file_name, message)

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
        finally:
            # Clear contextvar to avoid contaminating logs from other operations
            if job_id:
                set_job_id(None)


    def _check_max_images_limit(
        self,
        extract_process_images: bool,
        document_data: ExtractionResult[BaseFileMetadata],
        file_id: str,
        file_name: str,
        file_path_obj: Path,
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Checks if document exceeds max images per document limit.

        Returns:
            Tuple of (limit_exceeded, error_message, result_info).
            When limit exceeded: (True, error_message, result_info).
            When ok: (False, None, None).
        """
        if not extract_process_images or document_data.images is None:
            return False, None, None
        if len(document_data.images) <= self.config.max_images_per_document:
            return False, None, None
        image_count = len(document_data.images)
        limit = self.config.max_images_per_document
        error_message = (
            f"Document has too many images ({image_count}), maximum allowed is {limit}"
        )
        self.pipeline_logger.file_processing_error(file_id, file_name, error_message)
        result_info = {
            "file_id": file_id,
            "file_name": document_data.metadata.file_name,
            "file_path": str(file_path_obj)
        }
        return True, error_message, result_info

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