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
    Formats data according to the summary schema.
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
                - file_name: str (required)
                - source_id: str (optional, uses file_id if not provided)
                - type_file: str (required, typically "summary_<file_type>")
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
        if 'file_name' not in metadata:
            logger.error("metadata must contain 'file_name'")
            raise ValueError("metadata must contain 'file_name'")
        if 'type_file' not in metadata:
            logger.error("metadata must contain 'type_file'")
            raise ValueError("metadata must contain 'type_file'")

        # Get metadata values
        file_id = str(metadata['file_id'])
        file_name = str(metadata['file_name'])
        source_id = str(metadata.get('source_id', file_id))
        type_file = str(metadata['type_file'])
        total_pages = str(metadata.get('total_pages', metadata.get('num_pages', '0')))
        total_chapters = str(metadata.get('total_chapters', metadata.get('chapters', '')))
        total_num_image = str(metadata.get('total_num_image', metadata.get('num_image', '0')))


        # Get current date
        current_date = datetime.now().date().strftime('%Y-%m-%d')

        data = {
            "text": summary,
            "text_embedding": embedding,
            "image_embedding": "",
            "audio_embedding": "",
            "date": current_date,
            "source_id": source_id,
            "file_id": file_id,
            "file_name": file_name,
            "type_file": type_file,
            "total_pages": total_pages,
            "total_chapters": total_chapters,
            "total_num_image": total_num_image,
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

