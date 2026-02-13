"""
Summary preparer for Milvus
Prepares document summaries for insertion into Milvus
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from src.utils import get_logger


class SummaryPreparer:
    """
    Prepares document summaries for insertion into Milvus.
    Formats data according to the document schema.
    """

    @staticmethod
    def prepare(
        *,
        summary: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepares a document summary for insertion into Milvus.

        Args:
            summary: Summary text.
            embedding: Summary embedding vector.
            tokens: Token count (optional).
            metadata: Metadata dictionary with:
                - file_id: str (required)
                - file_type: str (required, typically "summary_<origin_file_type>")
                - file_name: str (required, typically "summary_<origin_file_name>")
                - total_pages: int or str (optional, default "0")
                - total_chapters: str (optional, default "")
                - total_num_image: int or str (optional, default "0")

        Returns:
            Dict[str, Any]: Prepared data dictionary ready for Milvus.

        Raises:
            ValueError: If required metadata is missing.
        """
        logger = get_logger(__name__)
        
        if not summary:
            logger.error("Summary cannot be empty")
            raise ValueError("summary cannot be empty")

        metadata = metadata or {}
        
        logger.debug(
            "Preparing summary",
            extra={
                "summary_length": len(summary),
                "file_id": metadata.get('file_id'),
                "file_name": metadata.get('file_name')
            }
        )
        
        # Validate required metadata
        if 'file_id' not in metadata:
            logger.error("metadata must contain 'file_id'")
            raise ValueError("metadata must contain 'file_id'")
        if 'file_type' not in metadata:
            logger.error("metadata must contain 'file_type'")
            raise ValueError("metadata must contain 'file_type'")
        if 'file_name' not in metadata:
            logger.error("metadata must contain 'file_name'")
            raise ValueError("metadata must contain 'file_name'")

        # Get metadata values
        file_id = str(metadata['file_id'])
        file_type = str(metadata['file_type'])
        file_name = str(metadata['file_name'])
        full_pages = str(metadata.get('full_pages', '0'))
        chapters = str(metadata.get('chapters', 'false'))
        full_images = str(metadata.get('full_images', '0'))
        current_date = datetime.now().date().strftime('%Y-%m-%d')

        data = {
            "file_id": file_id,
            "file_type": file_type,
            "file_name": file_name,
            "text": summary,
            "text_embedding": embedding,
            "pages": full_pages,
            "chapters": chapters,
            "image_number": "",
            "image_number_in_page": "",
            "full_images": full_images,
            "date": current_date,
        }

        logger.info(
            "Summary prepared successfully",
            extra={
                "file_id": file_id,
                "file_name": file_name,
                "summary_length": len(summary)
            }
        )

        return data

