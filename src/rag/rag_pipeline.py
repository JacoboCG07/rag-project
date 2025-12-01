"""
RAG Pipeline
Integrates DocumentExtractionManager with DocumentProcessor for end-to-end document processing
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import hashlib

from .extractors.base.types import ExtractionResult, BaseFileMetadata
from .extractors import DocumentExtractionManager
from .processing.processor import DocumentProcessor
from .config import RAGPipelineConfig


class RAGPipeline:
    """
    High-level pipeline that integrates DocumentExtractionManager with DocumentProcessor.
    Provides end-to-end document processing from folder to Milvus.
    """

    def __init__(
        self,
        *,
        config: RAGPipelineConfig
    ):
        """
        Initializes the RAG Pipeline.

        Args:
            config: RAGPipelineConfig with all configuration parameters.
            generate_embeddings_func: Function to generate embeddings (must receive text and return embedding).
            generate_summary_func: Function to generate summaries (optional, required if generate_summary=True in config).
        """
        self.config = config
        
        # Initialize DocumentProcessor with config
        self.document_processor = DocumentProcessor(
            dbname=config.milvus.dbname,
            collection_name_documents=config.collection_name_documents,
            collection_name_summaries=config.collection_name_summaries,
            generate_embeddings_func=config.generate_embeddings_func,
            generate_summary_func=config.generate_summary_func,
            alias=config.milvus.alias,
            embedding_dim=config.embedding_dim,
            uri=config.milvus.uri,
            token=config.milvus.token,
            host=config.milvus.host,
            port=config.milvus.port
        )
        
    def process_single_file(
        self,
        *,
        file_path: str,
        extract_process_images: Optional[bool] = None,
        partition_name: str

    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Processes a single file and inserts it into Milvus.

        Args:
            file_path: Path to the file to process.
            file_id: File ID (optional, generates from file_path if not provided).
            extract_images: Whether to extract images from PDFs (overrides config default).
            generate_summary: Whether to generate summary (overrides config default).

        Returns:
            Tuple[bool, str, Dict]: (success, message, result_info).
            result_info contains: file_id, file_name, file_path.
        """
        # Initialize DocumentExtractionManager with parent folder
        file_path_obj = Path(file_path)
        extraction_manager = DocumentExtractionManager(folder_path=str(file_path_obj.parent))
        file_id = self._generate_file_id(file_path)

        # Extract the document data
        #mandarle el file_id
        document_data: ExtractionResult[BaseFileMetadata] = extraction_manager.extract_file(
            file_path=file_path_obj,
            extract_images=extract_process_images
        )

        file_name = document_data.metadata.file_name

        # Process and insert document in milvus
        #mandarle el file_id
        success, message = self.document_processor.process_and_insert(
            file_name=file_name,
            document_data=document_data,
            process_images=extract_process_images,
            partition_name=partition_name
        )
        
        # Prepare result info
        result_info = {
            "file_id": file_id,
            "file_name": document_data.metadata.file_name,
            "file_path": str(file_path_obj)
        }
        
        return success, message, result_info


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
        self.document_processor.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()