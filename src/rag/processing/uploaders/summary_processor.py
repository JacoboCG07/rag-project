"""
Summary processor for Milvus
Handles generation and uploading of document summaries
"""

from typing import Dict, List, Any, Optional, Tuple, Callable

from rag.extractors.base import ExtractionResult
from ..milvus.milvus_client import MilvusClient

class SummaryProcessor:
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

    def process_and_upload_summary(
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
            # Validate data format
            if not isinstance(document_data, ExtractionResult):
                raise ValueError("document_data must be an ExtractionResult instance")

            content = document_data.content
            images = document_data.images or []
            metadata_obj = document_data.metadata

            if not content or not isinstance(content, list):
                raise ValueError("content must be a non-empty list of texts")

            # Get file_name and file_type from metadata
            file_name = metadata_obj.file_name
            file_type = metadata_obj.file_type if hasattr(metadata_obj, 'file_type') else 'document'
            source_id = file_id  # Use file_id as source_id
            num_images = len(images) if images else 0

            # Create partition if it doesn't exist
            self.milvus_client.create_partition(partition_name=partition_name)

            # Generate summary from content
            summary = self._generate_summary_from_content(content=content)

            if not summary:
                return False, f"Failed to generate summary for {file_name}"

            # Process and insert summary
            self._process_and_insert_summary(
                summary=summary,
                file_id=file_id,
                file_name=file_name,
                source_id=source_id,
                file_type=file_type,
                partition_name=partition_name,
                num_pages=len(content),
                num_images=num_images
            )

            return True, f"Summary for {file_name} generated and uploaded successfully"

        except Exception as e:
            # Get file_name from metadata if available for error message
            try:
                file_name = document_data.metadata.file_name if hasattr(document_data, 'metadata') else "unknown"
            except:
                file_name = "unknown"
            error_msg = f"Error processing summary for {file_name}: {str(e)}"
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
        # Combine all content into a single text
        full_text = "\n\n".join([
            text.strip() for text in content 
            if text and isinstance(text, str)
        ])
        
        if not full_text:
            return ""
        
        # Generate summary using the provided function
        summary = self.generate_summary_func(full_text)
        
        if not isinstance(summary, str):
            raise ValueError(
                "generate_summary_func must return a string"
            )
        
        return summary.strip()

    def _process_and_insert_summary(
        self,
        *,
        summary: str,
        file_id: str,
        file_name: str,
        source_id: str,
        file_type: str,
        partition_name: str,
        num_pages: int,
        num_images: int
    ) -> None:
        """
        Processes summary and inserts it into Milvus.

        Args:
            summary: Document summary text.
            file_id: File ID.
            file_name: File name.
            source_id: Source ID.
            file_type: File type.
            partition_name: Partition name.
            num_pages: Number of pages in the document.
            num_images: Number of images in the document.
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
            file_name=file_name,
            source_id=source_id,
            file_type=f"summary_{file_type}",
            num_pages=num_pages,
            num_images=num_images,
            pages=list(range(1, num_pages + 1)) if num_pages > 0 else []
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
        file_name: str,
        source_id: str,
        file_type: str,
        num_pages: int,
        num_images: int,
        pages: List[int]
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
            "file_name": file_name,
            "type_file": file_type,
            "total_pages": str(num_pages),
            "total_chapters": "",
            "total_num_image": str(num_images),
        }