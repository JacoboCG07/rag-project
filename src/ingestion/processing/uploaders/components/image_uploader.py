"""
Image uploader component
Handles image processing and insertion to Milvus
"""

from typing import List, Dict, Any, Tuple, Callable, Optional
from ...processors import ImageProcessor, validate_images_list
from ...milvus.milvus_client import MilvusClient
from ...preparers.document_preparer import DocumentPreparer
from src.utils import get_logger


class ImageUploader:
    """
    Handles image processing and uploading to Milvus.
    Responsible for: validation -> description -> embeddings -> insertion.
    """

    def __init__(
        self,
        *,
        milvus_client: MilvusClient,
        generate_embeddings_func: Callable[[str], Any],
        describe_image_func: Optional[Callable[[str], str]] = None,
    ):
        """
        Initializes the image uploader.

        Args:
            milvus_client: Milvus client for documents collection.
            generate_embeddings_func: Function to generate embeddings.
            describe_image_func: Function to describe image (optional).
        """
        self.milvus_client = milvus_client
        self.logger = get_logger(__name__)

        # Initialize processor
        self._image_processor = ImageProcessor(
            describe_image_func=describe_image_func,
            generate_embeddings_func=generate_embeddings_func,
        )

        self.logger.info(
            "Initializing ImageUploader",
            extra={"has_describe_image_func": describe_image_func is not None},
        )

    def process_and_upload(
        self,
        *,
        images: List[Any],
        file_id: str,
        file_name: str,
        file_type: str,
        partition_name: str,
    ) -> int:
        """
        Processes images and uploads them to Milvus.

        Args:
            images: List of images to process.
            file_id: Unique file ID.
            file_name: File name.
            file_type: File type.
            partition_name: Partition name for Milvus.

        Returns:
            int: Number of successfully processed images.
        """
        if not images:
            self.logger.debug(
                "No images to process",
                extra={"file_id": file_id},
            )
            return 0

        # Process images
        image_texts, image_embeddings, images_metadata = self._process_images(
            images=images,
            file_id=file_id,
        )

        processed_count = len(image_texts)

        # Insert to Milvus if any were processed successfully
        if image_texts:
            self._insert_images(
                file_id=file_id,
                file_name=file_name,
                file_type=file_type,
                image_texts=image_texts,
                image_embeddings=image_embeddings,
                images_metadata=images_metadata,
                partition_name=partition_name,
            )

        return processed_count

    def _process_images(
        self,
        *,
        images: List[Any],
        file_id: str,
    ) -> Tuple[List[str], List[List[float]], List[Dict[str, Any]]]:
        """
        Validates and processes images into descriptions and embeddings.

        Args:
            images: List of images (Pydantic models or dicts).
            file_id: File ID for logging purposes.

        Returns:
            Tuple[List[str], List[List[float]], List[Dict[str, Any]]]:
                (image_descriptions, embeddings, images_metadata).
        """
        # Convert ImageData Pydantic models to dicts for validation
        images_dict = [
            img.model_dump() if hasattr(img, "model_dump") else img for img in images
        ]

        # Validate image structure
        is_valid, error_msg = validate_images_list(images_dict)
        if not is_valid:
            self.logger.error(
                f"Invalid image structure: {error_msg}",
                extra={"file_id": file_id, "images_count": len(images_dict)},
            )
            raise ValueError(f"Invalid image structure: {error_msg}")

        # Process images using ImageProcessor
        image_texts, image_embeddings, images_metadata = self._image_processor.process(
            images=images_dict,
            file_id=file_id,
        )

        self.logger.debug(
            "Images processed",
            extra={
                "file_id": file_id,
                "input_images": len(images_dict),
                "processed_images": len(image_texts),
            },
        )

        return image_texts, image_embeddings, images_metadata

    def _insert_images(
        self,
        *,
        file_id: str,
        file_name: str,
        file_type: str,
        image_texts: List[str],
        image_embeddings: List[List[float]],
        images_metadata: List[Dict[str, Any]],
        partition_name: str,
    ) -> None:
        """
        Inserts image descriptions into Milvus.

        Args:
            file_id: Unique file ID.
            file_name: File name.
            file_type: File type.
            image_texts: Image descriptions.
            image_embeddings: Embeddings for each image description.
            images_metadata: Metadata for each image.
            partition_name: Partition name for Milvus.
        """
        file_metadata = {
            "file_id": file_id,
            "file_name": file_name,
            "type_file": f"image_{file_type}",
        }

        prepared_data = DocumentPreparer.prepare_images(
            image_descriptions=image_texts,
            embeddings=image_embeddings,
            file_metadata=file_metadata,
            images_metadata=images_metadata,
        )

        self.milvus_client.insert_prepared_data(
            prepared_data=prepared_data,
            partition_name=partition_name,
        )

        self.logger.info(
            "Images inserted successfully",
            extra={
                "file_id": file_id,
                "file_name": file_name,
                "images_count": len(image_texts),
                "partition_name": partition_name,
            },
        )

