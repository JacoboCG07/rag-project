"""
Summary preparer for Milvus
Prepares document summaries for insertion into Milvus
"""

from typing import Dict, Any, Optional, List

from src.utils import get_logger

from .summary_insert_dto import build_summary_insert_dict
from .milvus_insert_dto import SCHEMA_KEYS


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
                - full_pages: int or str (optional, default 0)
                - chapters: bool or str "true"/"false" (optional, default "false")
                - full_images: int or str (optional, default 0)

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
        full_pages_raw = metadata.get('full_pages', 0)
        num_pages = int(full_pages_raw) if full_pages_raw is not None else 0
        chapters_raw = metadata.get('chapters', 'false')
        has_chapters = (
            chapters_raw is True
            or (isinstance(chapters_raw, str) and chapters_raw.lower() == 'true')
        )
        full_images_raw = metadata.get('full_images', 0)
        num_images = int(full_images_raw) if full_images_raw is not None else 0

        # Log dropped keys (metadata fields not in schema)
        dropped = set(metadata.keys()) - SCHEMA_KEYS
        if dropped:
            logger.debug(
                "Fields not in schema, not added to insert",
                extra={
                    "dropped_keys": list(dropped),
                    "source": "metadata",
                    "file_id": file_id,
                },
            )

        data = build_summary_insert_dict(
            summary=summary,
            text_embedding=embedding,
            file_id=file_id,
            file_type=file_type,
            file_name=file_name,
            num_pages=num_pages,
            has_chapters=has_chapters,
            num_images=num_images,
        )

        logger.info(
            "Summary prepared successfully",
            extra={
                "file_id": file_id,
                "file_name": file_name,
                "summary_length": len(summary)
            }
        )

        return data

