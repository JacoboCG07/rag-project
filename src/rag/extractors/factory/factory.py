"""
Factory for creating extractors based on file type
"""
from pathlib import Path
from typing import List
from ..base import BaseDocumentExtractor
from ..pdf import PDFExtractor
from ..txt import TXTExtractor
from src.utils import get_logger


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
        logger = get_logger(__name__)
        file_extension = Path(file_path).suffix.lower()
        
        logger.debug(
            "Creating extractor for file",
            extra={
                "file_path": file_path,
                "file_extension": file_extension
            }
        )
        
        if file_extension == '.pdf':
            logger.debug("Creating PDFExtractor")
            return PDFExtractor(file_path)
        elif file_extension == '.txt':
            logger.debug("Creating TXTExtractor")
            return TXTExtractor(file_path)
        else:
            error_msg = f"Unsupported file type: {file_extension}"
            logger.error(
                error_msg,
                extra={
                    "file_path": file_path,
                    "file_extension": file_extension,
                    "supported_extensions": DocumentExtractorFactory._SUPPORTED_EXTENSIONS
                }
            )
            raise ValueError(error_msg)

