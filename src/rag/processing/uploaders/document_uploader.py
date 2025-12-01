"""
Document uploader for Milvus
Handles processing and uploading of full documents
"""

from ..milvus.milvus_client import MilvusClient
from typing import Dict, List, Any, Optional, Tuple, Callable
from ...extractors.base.types import ExtractionResult, BaseFileMetadata, ImageData
from .validators import validate_images_list

class DocumentUploader:
    """
    Handles processing and uploading of full documents to Milvus.
    Single responsibility: document processing and insertion.
    """

    def __init__(
        self,
        *,
        milvus_client: MilvusClient,
        generate_embeddings_func: Callable[[str], Any]
    ):
        """
        Initializes the document uploader.

        Args:
            milvus_client: Milvus client for documents collection.
            generate_embeddings_func: Function to generate embeddings (must receive text and return embedding).
        """
        self.milvus_client = milvus_client
        self.generate_embeddings_func = generate_embeddings_func

    def upload_document(
        self,
        *,
        document_data: ExtractionResult,
        process_images: bool = False,
        partition_name: str
    ) -> Tuple[bool, str]:
        """
        Processes and uploads a document to Milvus.

        Args:
            document_data: ExtractionResult with 'content' (list of texts), 'images' (optional list of ImageData), 
                          and 'metadata' (BaseFileMetadata or subclass).
            file_id: Unique file ID.
            file_name: File name (optional, will use metadata.file_name if available, else file_id).
            source_id: Source ID (optional, uses file_id if not provided).
            file_type: File type (optional, will use metadata.file_type if available, else 'document').
            process_images: Whether to process and vectorize images (default False).

        Returns:
            Tuple[bool, str]: (success, message).

        Raises:
            ValueError: If document_data doesn't have the expected format.
        """
        try:
            # Validate data format
            if not isinstance(document_data, ExtractionResult):
                raise ValueError("document_data must be an ExtractionResult instance")

            content = document_data.content
            images = document_data.images or []
            metadata = document_data.metadata

            if not content:
                raise ValueError("document_data must contain 'content' with at least one element")

            if not isinstance(content, list):
                raise ValueError("'content' must be a list of texts")

            self.milvus_client.create_partition(partition_name=partition_name)

            # Process texts
            texts, embeddings, tokens = self._process_texts(content=content)

            # Prepare metadata
            metadata = self._prepare_metadata(
                file_id=file_id,
                file_name=file_name,
                source_id=source_id or file_id,
                file_type=file_type,
                num_pages=len(content),
                num_images=len(images),
                pages=list(range(1, len(content) + 1))
            )

            # Insert texts
            self.milvus_client.insert_documents(
                texts=texts,
                embeddings=embeddings,
                tokens=tokens,
                metadata=metadata,
                partition_name=partition_name
            )

            # Process images if requested and they exist
            if process_images and images:
                # Convert ImageData Pydantic models to dicts for validation
                images_dict = [img.model_dump() if hasattr(img, 'model_dump') else img for img in images]
                # Validate image structure
                is_valid, error_msg = validate_images_list(images_dict)
                if not is_valid:
                    raise ValueError(f"Invalid image structure: {error_msg}")
                
                self._process_images(
                    images=images_dict,  # Pass as dicts for processing
                    file_id=file_id,
                    file_name=file_name,
                    file_type=file_type,
                    partition_name=partition_name
                )

            message = f"Document {file_name} uploaded successfully"
            if process_images and images:
                message += f" (with {len(images)} images processed)"

            return True, message

        except Exception as e:
            error_msg = f"Error uploading document {file_name}: {str(e)}"
            return False, error_msg

    def _process_texts(
        self,
        *,
        content: List[str]
    ) -> Tuple[List[str], List[List[float]], List[str]]:
        """
        Processes texts and generates embeddings.

        Args:
            content: List of texts.

        Returns:
            Tuple[List[str], List[List[float]], List[str]]: (texts, embeddings, tokens).
        """
        texts = []
        embeddings = []
        tokens = []

        for text in content:
            if not text or not isinstance(text, str):
                continue

            # Clean text
            cleaned_text = text.strip()
            if not cleaned_text:
                continue

            texts.append(cleaned_text)

            # Generate embedding
            embedding = self.generate_embeddings_func(cleaned_text)
            if isinstance(embedding, tuple):
                # If it returns (embedding, tokens)
                embedding, token_count = embedding
                tokens.append(str(token_count))
            else:
                tokens.append("")
            embeddings.append(embedding)

        return texts, embeddings, tokens

    def _process_images(
        self,
        *,
        images: List[Any],
        file_id: str,
        file_name: str,
        source_id: str,
        file_type: str,
        partition_name: str
    ) -> None:
        """
        Processes images and inserts them into Milvus.

        Args:
            images: List of images (can be paths, base64, etc.).
            file_id: File ID.
            file_name: File name.
            source_id: Source ID.
            file_type: File type.
            partition_name: Partition name.
        """
        if not images:
            return

        image_texts = []
        image_embeddings = []
        image_tokens = []

        for image in images:
            # Images are validated before calling this method, so we know they have the expected structure
            # Create description from image metadata (page, image_number_in_page, etc.)
            # Format: "Image {image_number} (page {page}, image {image_number_in_page}) from {file_name}"
            page = image.get('page', 0)
            image_num_in_page = image.get('image_number_in_page', 0)
            image_num = image.get('image_number', 0)
            
            image_description = f"Image {image_num} (page {page}, image {image_num_in_page}) from {file_name}"
            
            image_texts.append(image_description)

            # Generate embedding from image description
            embedding = self.generate_embeddings_func(image_description)
            if isinstance(embedding, tuple):
                embedding, token_count = embedding
                image_tokens.append(str(token_count))
            else:
                image_tokens.append("")
            image_embeddings.append(embedding)

        # Prepare metadata for images
        metadata = self._prepare_metadata(
            file_id=file_id,
            file_name=file_name,
            source_id=source_id,
            file_type=f"image_{file_type}",
            num_pages=len(images),
            num_images=len(images),
            pages=list(range(1, len(images) + 1))
        )

        # Insert images
        self.milvus_client.insert_documents(
            texts=image_texts,
            embeddings=image_embeddings,
            tokens=image_tokens,
            metadata=metadata,
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
            "source_id": source_id,
            "file_id": file_id,
            "file_name": file_name,
            "type_file": file_type,
            "pages": [str(p) for p in pages],
            "num_image": str(num_images),
        }

