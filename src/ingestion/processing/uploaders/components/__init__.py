"""
Components for document uploading
Specialized uploaders for text and image processing
"""

from .text_uploader import TextUploader
from .image_uploader import ImageUploader

__all__ = [
    'TextUploader',
    'ImageUploader',
]

