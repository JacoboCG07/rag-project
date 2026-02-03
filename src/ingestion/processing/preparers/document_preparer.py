"""
Document preparer for Milvus
Prepares document chunks for insertion into Milvus
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from src.utils import get_logger


class DocumentPreparer:
    """
    Prepares document chunks for insertion into Milvus.
    Formats data according to the document schema.
    """

    @staticmethod
    def prepare(
        *,
        texts: List[str],
        embeddings: List[List[float]],
        file_metadata: Dict[str, Any],
        chunks_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepares document chunks for insertion into Milvus.
        Separates file-level metadata from chunk-level metadata.

        Args:
            texts: List of text chunks.
            embeddings: List of corresponding embeddings.
            file_metadata: File-level metadata (common to all chunks) with:
                - file_id: str (required)
                - file_name: str (required)
                - source_id: str (optional, uses file_id if not provided)
                - type_file: str (required)
            chunks_metadata: List of chunk-level metadata dictionaries (one per chunk) with:
                - pages: int, str, or List[int] (optional)
                - chapters: str or List[str] (optional)
                - image_number: int or str (optional, default "")
                - image_number_in_page: int or str (optional, default "")
                If None, creates empty metadata for each chunk.

        Returns:
            List[Dict[str, Any]]: List of prepared data dictionaries ready for Milvus.

        Raises:
            ValueError: If required metadata is missing or data lengths don't match.
        """
        logger = get_logger(__name__)
        
        if len(texts) != len(embeddings):
            error_msg = f"Number of texts ({len(texts)}) must match number of embeddings ({len(embeddings)})"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not texts:
            logger.debug("Empty texts list, returning empty result")
            return []

        logger.debug(
            "Preparing document chunks",
            extra={
                "texts_count": len(texts),
                "file_id": file_metadata.get('file_id'),
                "file_name": file_metadata.get('file_name'),
                "has_chunks_metadata": chunks_metadata is not None
            }
        )

        # Validate required file metadata
        if 'file_id' not in file_metadata:
            logger.error("file_metadata must contain 'file_id'")
            raise ValueError("file_metadata must contain 'file_id'")
        if 'file_name' not in file_metadata:
            logger.error("file_metadata must contain 'file_name'")
            raise ValueError("file_metadata must contain 'file_name'")
        if 'type_file' not in file_metadata:
            logger.error("file_metadata must contain 'type_file'")
            raise ValueError("file_metadata must contain 'type_file'")

        # Get file-level metadata values
        file_id = str(file_metadata['file_id'])
        file_name = str(file_metadata['file_name'])
        source_id = str(file_metadata.get('source_id', file_id))
        type_file = str(file_metadata['type_file'])

        # Initialize chunks_metadata if not provided
        if chunks_metadata is None:
            chunks_metadata = [{}] * len(texts)
        elif len(chunks_metadata) != len(texts):
            raise ValueError(
                f"Number of chunks_metadata ({len(chunks_metadata)}) must match "
                f"number of texts ({len(texts)})"
            )

        # Get current date
        current_date = datetime.now().date().strftime('%Y-%m-%d')

        # Prepare data for each chunk
        prepared_data = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            chunk_meta = chunks_metadata[i] if i < len(chunks_metadata) else {}

            # Get pages for this chunk
            pages_value = chunk_meta.get('pages', "")
            if isinstance(pages_value, list):
                pages_str = ','.join(str(p) for p in pages_value)
            elif pages_value:
                pages_str = str(pages_value)
            else:
                pages_str = ""

            # Get chapters for this chunk
            chapters_value = chunk_meta.get('chapters', "")
            if isinstance(chapters_value, list):
                chapters_str = ','.join(str(c) for c in chapters_value)
            elif chapters_value:
                chapters_str = str(chapters_value)
            else:
                chapters_str = ""

            # Get image_number for this chunk
            image_number = chunk_meta.get('image_number', "")
            image_number_str = str(image_number) if image_number else ""

            # Get image_number_in_page for this chunk
            image_number_in_page = chunk_meta.get('image_number_in_page', "")
            image_number_in_page_str = str(image_number_in_page) if image_number_in_page else ""

            data = {
                "text": text,
                "text_embedding": embedding,
                "date": current_date,
                "source_id": source_id,
                "file_id": file_id,
                "file_name": file_name,
                "type_file": type_file,
                "pages": pages_str,
                "chapters": chapters_str,
                "image_number": image_number_str,
                "image_number_in_page": image_number_in_page_str,
            }

            prepared_data.append(data)

        logger.info(
            "Image descriptions prepared successfully",
            extra={
                "images_count": len(prepared_data),
                "file_id": file_id,
                "file_name": file_name
            }
        )

        return prepared_data

    @staticmethod
    def prepare_images(
        *,
        image_descriptions: List[str],
        embeddings: List[List[float]],
        file_metadata: Dict[str, Any],
        images_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepares image descriptions for insertion into Milvus.
        Separates file-level metadata from image-level metadata.

        Args:
            image_descriptions: List of image descriptions.
            embeddings: List of corresponding embeddings.
            file_metadata: File-level metadata (common to all images) with:
                - file_id: str (required)
                - file_name: str (required)
                - source_id: str (optional, uses file_id if not provided)
                - type_file: str (required, typically "image_<file_type>")
            images_metadata: List of image-level metadata dictionaries (one per image) with:
                - pages: int or str (optional)
                - image_number: int or str (optional, default "")
                - image_number_in_page: int or str (optional, default "")
                If None, creates empty metadata for each image.

        Returns:
            List[Dict[str, Any]]: List of prepared data dictionaries ready for Milvus.

        Raises:
            ValueError: If required metadata is missing or data lengths don't match.
        """
        logger = get_logger(__name__)
        
        if len(image_descriptions) != len(embeddings):
            error_msg = (
                f"Number of descriptions ({len(image_descriptions)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not image_descriptions:
            logger.debug("Empty image descriptions list, returning empty result")
            return []

        logger.debug(
            "Preparing image descriptions",
            extra={
                "images_count": len(image_descriptions),
                "file_id": file_metadata.get('file_id'),
                "file_name": file_metadata.get('file_name'),
                "has_images_metadata": images_metadata is not None
            }
        )

        # Validate required file metadata
        if 'file_id' not in file_metadata:
            logger.error("file_metadata must contain 'file_id'")
            raise ValueError("file_metadata must contain 'file_id'")
        if 'file_name' not in file_metadata:
            logger.error("file_metadata must contain 'file_name'")
            raise ValueError("file_metadata must contain 'file_name'")
        if 'type_file' not in file_metadata:
            logger.error("file_metadata must contain 'type_file'")
            raise ValueError("file_metadata must contain 'type_file'")

        # Get file-level metadata values
        file_id = str(file_metadata['file_id'])
        file_name = str(file_metadata['file_name'])
        source_id = str(file_metadata.get('source_id', file_id))
        type_file = str(file_metadata['type_file'])

        # Initialize images_metadata if not provided
        if images_metadata is None:
            images_metadata = [{}] * len(image_descriptions)
        elif len(images_metadata) != len(image_descriptions):
            raise ValueError(
                f"Number of images_metadata ({len(images_metadata)}) must match "
                f"number of image_descriptions ({len(image_descriptions)})"
            )

        # Get current date
        current_date = datetime.now().date().strftime('%Y-%m-%d')

        # Prepare data for each image
        prepared_data = []
        for i, (description, embedding) in enumerate(zip(image_descriptions, embeddings)):
            image_meta = images_metadata[i] if i < len(images_metadata) else {}

            # Get page for this image
            pages_value = image_meta.get('pages', "")
            pages_str = str(pages_value) if pages_value else ""

            # Get image_number for this image
            image_number = image_meta.get('image_number', "")
            image_number_str = str(image_number) if image_number else ""

            # Get image_number_in_page for this image
            image_number_in_page = image_meta.get('image_number_in_page', "")
            image_number_in_page_str = str(image_number_in_page) if image_number_in_page else ""

            data = {
                "text": description,
                "text_embedding": embedding,
                "image_embedding": "",
                "audio_embedding": "",
                "date": current_date,
                "source_id": source_id,
                "file_id": file_id,
                "file_name": file_name,
                "type_file": type_file,
                "pages": pages_str,
                "chapters": "",
                "image_number": image_number_str,
                "image_number_in_page": image_number_in_page_str,
            }

            prepared_data.append(data)

        return prepared_data

