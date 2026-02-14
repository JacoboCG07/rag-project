"""
Text uploader component
Handles text processing and insertion to Milvus
"""

from typing import List, Dict, Any, Tuple, Callable
from ...processors import TextProcessor, ChunkProcessor
from ...milvus.milvus_client import MilvusClient
from ...preparers.document_preparer import DocumentPreparer
from src.utils import get_logger


class TextUploader:
    """
    Handles text processing and uploading to Milvus.
    Responsible for: chunking -> embeddings -> insertion.
    """

    def __init__(
        self,
        *,
        milvus_client: MilvusClient,
        generate_embeddings_func: Callable[[str], Any],
        chunk_size: int = 2000,
        chunk_overlap: int = 0,
        detect_chapters: bool = True,
    ):
        """
        Initializes the text uploader.

        Args:
            milvus_client: Milvus client for documents collection.
            generate_embeddings_func: Function to generate embeddings.
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
            detect_chapters: Whether to detect chapters in text.
        """
        self.milvus_client = milvus_client
        self.logger = get_logger(__name__)

        # Initialize processors
        self._text_processor = TextProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            detect_chapters=detect_chapters,
        )
        self._chunk_processor = ChunkProcessor(
            generate_embeddings_func=generate_embeddings_func
        )

        self.logger.info(
            "Initializing TextUploader",
            extra={
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "detect_chapters": detect_chapters,
            },
        )

    def process_and_upload(
        self,
        *,
        content: List[str],
        file_id: str,
        file_name: str,
        file_type: str,
        partition_name: str,
    ) -> Tuple[List[str], int]:
        """
        Processes texts and uploads them to Milvus.

        Args:
            content: List of text pages.
            file_id: Unique file ID.
            file_name: File name.
            file_type: File type.
            partition_name: Partition name for Milvus.

        Returns:
            Tuple[List[str], int]: (chunks, chunks_count)
        """
        # Process texts
        chunks, embeddings, chunks_metadata = self._process_texts(content=content)

        self.logger.debug(
            "Texts processed",
            extra={
                "file_id": file_id,
                "input_pages": len(content),
                "output_chunks": len(chunks),
                "embeddings_count": len(embeddings),
            },
        )

        # Insert to Milvus
        self._insert_chunks(
            file_id=file_id,
            file_name=file_name,
            file_type=file_type,
            chunks=chunks,
            embeddings=embeddings,
            chunks_metadata=chunks_metadata,
            partition_name=partition_name,
        )

        return chunks, len(chunks)

    def _process_texts(
        self, *, content: List[str]
    ) -> Tuple[List[str], List[List[float]], List[Dict[str, Any]]]:
        """
        Processes texts into chunks and generates embeddings.

        Args:
            content: List of text pages.

        Returns:
            Tuple[List[str], List[List[float]], List[Dict]]: 
                (chunks, embeddings, chunks_metadata).
        """
        # Step 1: Chunk pages using TextProcessor
        chunks, chunks_metadata = self._text_processor.process_to_chunks(pages=content)

        # Step 2: Generate embeddings for chunks using ChunkProcessor
        chunks, embeddings, chunks_metadata = self._chunk_processor.process_to_embeddings(
            chunks=chunks,
            chunks_metadata=chunks_metadata,
            max_acceptable_loss=0.10,
        )

        return chunks, embeddings, chunks_metadata

    def _insert_chunks(
        self,
        *,
        file_id: str,
        file_name: str,
        file_type: str,
        chunks: List[str],
        embeddings: List[List[float]],
        chunks_metadata: List[Dict[str, Any]],
        partition_name: str,
    ) -> None:
        """
        Inserts text chunks into Milvus.

        Args:
            file_id: Unique file ID.
            file_name: File name.
            file_type: File type.
            chunks: Text chunks.
            embeddings: Embeddings for each chunk.
            chunks_metadata: Metadata for each chunk.
            partition_name: Partition name for Milvus.
        """
        file_metadata = {
            "file_id": file_id,
            "file_name": file_name,
            "type_file": file_type,
        }

        prepared_data = DocumentPreparer.prepare(
            texts=chunks,
            embeddings=embeddings,
            file_metadata=file_metadata,
            chunks_metadata=chunks_metadata,
        )

        self.milvus_client.create_partition(partition_name=partition_name)
        self.milvus_client.insert_prepared_data(
            prepared_data=prepared_data,
            partition_name=partition_name,
        )

        self.logger.info(
            "Document chunks inserted successfully",
            extra={
                "file_id": file_id,
                "file_name": file_name,
                "chunks_count": len(chunks),
                "partition_name": partition_name,
            },
        )

