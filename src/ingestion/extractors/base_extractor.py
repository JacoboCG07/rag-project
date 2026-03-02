"""
Abstract base class for document extractors
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from ..types import BaseFileMetadata, ExtractionResult


# Type variable for metadata types
MetadataType = TypeVar('MetadataType', bound=BaseFileMetadata)


class BaseDocumentExtractor(ABC, Generic[MetadataType]):
    """
    Abstract base class for document extractors
    
    Generic type parameter:
        MetadataType: Type of metadata this extractor returns (must extend BaseFileMetadata)
    """
    
    def __init__(self, file_path: str):
        """
        Initializes the document extractor
        
        Args:
            file_path: Path to the file to process
        """
        self.file_path = Path(file_path)
        self.file_name = self.file_path.name
        self.file_type = self.file_path.suffix.lower()
    
    @abstractmethod
    def extract(self, extract_images: bool = False) -> ExtractionResult[MetadataType]:
        """
        Extracts content from the document
        
        Args:
            extract_images: If True, also extracts images from the document
        
        Returns:
            ExtractionResult with extracted content, images, and metadata
        """
        raise NotImplementedError
    
    def get_metadata(self) -> BaseFileMetadata:
        """
        Gets basic file metadata
        
        Returns:
            BaseFileMetadata with file name and type
        """
        return BaseFileMetadata(
            file_name=self.file_name,
            file_type=self.file_type
        )


