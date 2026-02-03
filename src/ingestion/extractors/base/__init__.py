"""
Base module for document extractors
"""
from .base_extractor import BaseDocumentExtractor
from .types import (
    ImageData,
    BaseFileMetadata,
    PDFFileMetadata,
    ExtractionResult
)

__all__ = [
    'BaseDocumentExtractor',
    'ImageData',
    'BaseFileMetadata',
    'PDFFileMetadata',
    'ExtractionResult'
]

