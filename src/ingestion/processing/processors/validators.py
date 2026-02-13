"""
Validators for document processing
"""
from typing import Dict, Any, List, Tuple


def validate_image_structure(image: Any) -> bool:
    """
    Validates that an image has the expected structure from PDFExtractor.

    Expected structure:
    {
        'page': int,
        'image_number_in_page': int,
        'image_number': int,
        'image_base64': str,
        'image_format': str
    }

    Args:
        image: Image object to validate (should be a dict).

    Returns:
        bool: True if image has valid structure, False otherwise.
    """
    if not isinstance(image, dict):
        return False

    required_fields = [
        "page",
        "image_number_in_page",
        "image_number",
        "image_base64",
        "image_format",
    ]

    # Check that all required fields are present
    for field in required_fields:
        if field not in image:
            return False

    # Validate field types
    if not isinstance(image["page"], int) or image["page"] < 1:
        return False

    if not isinstance(image["image_number_in_page"], int) or image[
        "image_number_in_page"
    ] < 1:
        return False

    if not isinstance(image["image_number"], int) or image["image_number"] < 1:
        return False

    if not isinstance(image["image_base64"], str) or not image["image_base64"]:
        return False

    if not isinstance(image["image_format"], str) or not image["image_format"]:
        return False

    return True


def validate_images_list(images: List[Any]) -> Tuple[bool, str]:
    """
    Validates a list of images.

    Args:
        images: List of images to validate.

    Returns:
        Tuple[bool, str]: (is_valid, error_message).
        If is_valid is True, error_message is empty string.
    """
    if not isinstance(images, list):
        return False, "images must be a list"

    for idx, image in enumerate(images):
        if not validate_image_structure(image):
            return (
                False,
                "Image at index "
                f"{idx} has invalid structure. Expected dict with fields: page, "
                "image_number_in_page, image_number, image_base64, image_format",
            )

    return True, ""


