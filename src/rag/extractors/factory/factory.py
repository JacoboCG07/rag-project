"""
Factory for creating extractors based on file type
"""
from pathlib import Path
from typing import List
from ..base import BaseDocumentExtractor
from ..pdf import PDFExtractor
from ..txt import TXTExtractor


class DocumentExtractorFactory:
    """Factory for creating extractors based on file type"""
    
    # Supported file extensions (must match extractors available)
    _SUPPORTED_EXTENSIONS = ['.pdf', '.txt']
    
    @staticmethod
    def get_supported_extensions() -> List[str]:
        """
        Returns the list of supported file extensions.
        
        Returns:
            List of supported file extensions (e.g., ['.pdf', '.txt'])
        """
        return DocumentExtractorFactory._SUPPORTED_EXTENSIONS.copy()
    
    @staticmethod
    def create_extractor(file_path: str) -> BaseDocumentExtractor:
        """
        Creates an appropriate extractor based on the file extension
        
        Args:
            file_path: Path to the file
            
        Returns:
            Instance of the appropriate extractor
            
        Raises:
            ValueError: If the file type is not supported
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return PDFExtractor(file_path)
        elif file_extension == '.txt':
            return TXTExtractor(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

