"""
Summary processor for Milvus
Handles generation and uploading of document summaries
"""

from typing import Dict, List, Any, Optional, Tuple, Callable

from src.utils import get_logger

from ...types import ExtractionResult, MetadataType
from ..milvus.milvus_client import MilvusClient

class SummaryUploader:
    """
    Handles generation and uploading of document summaries to Milvus.
    Single responsibility: summary generation and insertion.
    """

    def __init__(
        self,
        *,
        milvus_client: MilvusClient,
        generate_summary_func: Callable[[str], str],
        generate_embeddings_func: Callable[[str], Any]
    ):
        """
        Initializes the summary processor.

        Args:
            milvus_client: Milvus client for summaries collection.
            generate_summary_func: Function to generate summary (must receive full text and return summary string).
            generate_embeddings_func: Function to generate embeddings (must receive text and return embedding).
        """
        self.milvus_client = milvus_client
        self.generate_summary_func = generate_summary_func
        self.generate_embeddings_func = generate_embeddings_func
        self.logger = get_logger(__name__)
        
        self.logger.info("Initializing SummaryProcessor")

    def upload_summary(
        self,
        *,
        document_data: ExtractionResult,
        file_id: str,
        partition_name: str
    ) -> Tuple[bool, str]:
        """
        Generates summary from content and uploads it to Milvus.

        Args:
            document_data: ExtractionResult with 'content' (list of texts), 'images' (optional list of ImageData), 
                          and 'metadata' (BaseFileMetadata or subclass).
            file_id: Unique file ID.
            partition_name: Partition name for Milvus.

        Returns:
            Tuple[bool, str]: (success, message).

        Raises:
            ValueError: If content is empty or invalid.
        """
        try:
            self.logger.info(
                "Starting summary processing",
                extra={
                    "file_id": file_id,
                    "partition_name": partition_name
                }
            )
            
            # Validate data format
            if not isinstance(document_data, ExtractionResult):
                self.logger.error("document_data must be an ExtractionResult instance")
                raise ValueError("document_data must be an ExtractionResult instance")

            content = document_data.content
            images = document_data.images or []
            metadata_obj: MetadataType = document_data.metadata

            if not content or not isinstance(content, list):
                self.logger.error("content must be a non-empty list of texts")
                raise ValueError("content must be a non-empty list of texts")

            # Get file_name, file_type, and chapters from metadata
            file_name = metadata_obj.file_name
            file_type = metadata_obj.file_type if hasattr(metadata_obj, 'file_type') else 'document'
            chapters = getattr(metadata_obj, 'chapters', False)

            num_images = len(images) if images else 0

            self.logger.debug(
                "Processing summary",
                extra={
                    "file_id": file_id,
                    "file_name": file_name,
                    "content_chunks": len(content),
                    "images_count": num_images
                }
            )

            # Create partition if it doesn't exist
            self.milvus_client.create_partition(partition_name=partition_name)

            # Generate summary from content
            summary = self._generate_summary_from_content(content=content)

            if not summary:
                self.logger.error("Failed to generate summary", extra={"file_id": file_id, "file_name": file_name})
                return False, f"Failed to generate summary for {file_name}"

            # Process and insert summary
            self._process_and_insert_summary(
                summary=summary,
                file_id=file_id,
                file_type=file_type,
                file_name=file_name,
                num_pages=len(content),
                chapters=chapters,
                num_images=num_images,
                partition_name=partition_name,
            )

            self.logger.info(
                "Summary generated and uploaded successfully",
                extra={
                    "file_id": file_id,
                    "file_name": file_name,
                    "summary_length": len(summary),
                    "partition_name": partition_name
                }
            )

            return True, f"Summary for {file_name} generated and uploaded successfully"

        except Exception as e:
            # Get file_name from metadata if available for error message
            try:
                file_name = document_data.metadata.file_name if hasattr(document_data, 'metadata') else "unknown"
            except:
                file_name = "unknown"
            error_msg = f"Error processing summary for {file_name}: {str(e)}"
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

    def _generate_summary_from_content(
        self,
        *,
        content: List[str]
    ) -> str:
        """
        Generates a summary from document content.

        Args:
            content: List of texts (pages) from the document.

        Returns:
            str: Generated summary.
        """
        self.logger.debug(
            "Generating summary from content",
            extra={"content_chunks": len(content)}
        )
        
        # Combine all content into a single text
        full_text = "\n\n".join([
            text.strip() for text in content 
            if text and isinstance(text, str)
        ])
        
        if not full_text:
            self.logger.warning("Empty full text after combining content chunks")
            return ""
        
        # Generate summary using the provided function
        summary = self.generate_summary_func(full_text)
        
        if not isinstance(summary, str):
            self.logger.error("generate_summary_func must return a string")
            raise ValueError(
                "generate_summary_func must return a string"
            )
        
        self.logger.debug(
            "Summary generated",
            extra={
                "original_length": len(full_text),
                "summary_length": len(summary.strip())
            }
        )
        
        return summary.strip()

    def _process_and_insert_summary(
        self,
        *,
        summary: str,
        file_id: str,
        file_type: str,
        file_name: str,
        num_pages: int,
        chapters: bool,
        num_images: int,
        partition_name: str,
    ) -> None:
        """
        Processes summary and inserts it into Milvus.

        Args:
            summary: Document summary text.
            file_id: File ID.
            file_type: File type.
            file_name: File name.
            num_pages: Number of pages in the document.
            num_images: Number of images in the document.
            partition_name: Partition name.
        """
        if not summary or not isinstance(summary, str):
            return

        # Clean summary text
        cleaned_summary = summary.strip()
        if not cleaned_summary:
            return

        # Generate embedding for summary
        embedding = self.generate_embeddings_func(cleaned_summary)
        if isinstance(embedding, tuple):
            # If it returns (embedding, token_count), extract just the embedding
            embedding, _ = embedding
        embeddings = [embedding]

        # Prepare metadata for summary
        metadata_summary = self._prepare_metadata(
            file_id=file_id,
            file_type=f"summary_{file_type}",
            file_name=f"summary_{file_name}",
            num_pages=num_pages,
            chapters=chapters,
            num_images=num_images,
        )

        # Insert summary into summaries collection
        self.milvus_client.insert_documents(
            texts=[cleaned_summary],
            embeddings=embeddings,
            metadata=metadata_summary,
            partition_name=partition_name
        )

    @staticmethod
    def _prepare_metadata(
        *,
        file_id: str,
        file_type: str,
        file_name: str,
        num_pages: int,
        chapters: bool,
        num_images: int,
    ) -> Dict[str, Any]:
        """
        Prepares metadata for insertion into Milvus.

        Args:
            file_id: File ID.
            file_name: File name.
            source_id: Source ID.
            file_type: File type.
            num_pages: Number of pages.
            num_images: Number of images.
            pages: List of page numbers.

        Returns:
            Dict: Prepared metadata.
        """
        return {
            "file_id": file_id,
            "file_type": file_type,
            "file_name": file_name,
            "full_pages": str(num_pages),
            "chapters": str(chapters),
            "full_images": str(num_images),
        }