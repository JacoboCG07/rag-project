"""
Image processor for document images
Handles image description and embedding generation
"""

from typing import List, Dict, Tuple, Any, Callable, Optional
from src.utils import get_logger


class ImageProcessor:
    """
    Processes images to generate descriptions and embeddings.
    Responsible for describing images and generating embeddings from descriptions.
    """

    def __init__(
        self,
        *,
        describe_image_func: Optional[Callable[[str], str]] = None,
        generate_embeddings_func: Callable[[str], Any],
    ):
        """
        Initializes the image processor.

        Args:
            describe_image_func: Function to describe image (must receive base64 image and return description string). Optional.
            generate_embeddings_func: Function to generate embeddings (must receive text and return embedding).
        """
        self.describe_image_func = describe_image_func
        self.generate_embeddings_func = generate_embeddings_func
        self.logger = get_logger(__name__)

        self.logger.info(
            "Initializing ImageProcessor",
            extra={
                "has_describe_image_func": describe_image_func is not None,
            },
        )

    def process(
        self,
        *,
        images: List[Dict[str, Any]],
        file_id: str,
    ) -> Tuple[List[str], List[List[float]], List[Dict[str, Any]]]:
        """
        Processes images to generate descriptions and embeddings.

        Args:
            images: List of image dictionaries with expected structure:
                - page: int
                - image_number_in_page: int
                - image_number: int
                - image_base64: str
            file_id: File ID for logging purposes.

        Returns:
            Tuple[List[str], List[List[float]], List[Dict[str, Any]]]: 
                (image_descriptions, embeddings, images_metadata).
                Only returns images that were successfully processed.
        """
        if not images:
            self.logger.debug("Empty images list provided, returning empty result")
            return [], [], []

        self.logger.debug(
            "Starting image processing",
            extra={
                "file_id": file_id,
                "images_count": len(images),
            },
        )

        image_texts: List[str] = []
        image_embeddings: List[List[float]] = []
        images_metadata: List[Dict[str, Any]] = []
        skipped_count = 0

        for image in images:
            # Images are validated before calling this method, so we know they have the expected structure
            page = image.get("page", 0)
            image_num_in_page = image.get("image_number_in_page", 0)
            image_num = image.get("image_number", 0)
            image_base64 = image.get("image_base64", "")

            # Only process image if we have describe_image_func and image_base64
            if not self.describe_image_func or not image_base64:
                # Skip this image if we can't describe it
                skipped_count += 1
                self.logger.debug(
                    "Skipping image (no describe function or base64)",
                    extra={
                        "file_id": file_id,
                        "page": page,
                        "image_number": image_num,
                    },
                )
                continue

            # Try to describe the image using LLM
            try:
                image_description = self.describe_image_func(image=image_base64)
            except Exception as e:
                # If description fails, skip this image and continue with the next one
                skipped_count += 1
                self.logger.warning(
                    f"Failed to describe image: {str(e)}",
                    extra={
                        "file_id": file_id,
                        "page": page,
                        "image_number": image_num,
                        "error_type": type(e).__name__,
                    },
                )
                continue

            image_texts.append(image_description)

            # Generate embedding from image description
            try:
                embedding = self.generate_embeddings_func(image_description)
                if isinstance(embedding, tuple):
                    # If it returns (embedding, token_count), extract just the embedding
                    embedding, _ = embedding
                image_embeddings.append(embedding)
            except Exception as e:
                # If embedding generation fails, remove the description we just added
                image_texts.pop()
                skipped_count += 1
                self.logger.warning(
                    f"Failed to generate embedding for image description: {str(e)}",
                    extra={
                        "file_id": file_id,
                        "page": page,
                        "image_number": image_num,
                        "error_type": type(e).__name__,
                    },
                )
                continue

            # Store metadata for this successfully processed image
            images_metadata.append(
                {
                    "pages": page,
                    "image_number": image_num,
                    "image_number_in_page": image_num_in_page,
                }
            )

        if skipped_count > 0:
            self.logger.debug(
                "Some images were skipped during processing",
                extra={
                    "file_id": file_id,
                    "total_images": len(images),
                    "processed_images": len(image_texts),
                    "skipped_images": skipped_count,
                },
            )

        self.logger.info(
            "Image processing completed",
            extra={
                "file_id": file_id,
                "input_images": len(images),
                "processed_images": len(image_texts),
                "skipped_images": skipped_count,
            },
        )

        return image_texts, image_embeddings, images_metadata


