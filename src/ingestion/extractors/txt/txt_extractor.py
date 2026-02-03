"""
Text file extractor
"""
from typing import List
from ..base import (
    BaseDocumentExtractor,
    BaseFileMetadata,
    ExtractionResult
)
from src.utils import get_logger


class TXTExtractor(BaseDocumentExtractor[BaseFileMetadata]):
    """Extractor for text files"""
    
    def __init__(self, file_path: str):
        """
        Initializes the TXT extractor
        
        Args:
            file_path: Path to the text file
        """
        super().__init__(file_path)
        self.logger = get_logger(__name__)
        self.logger.debug(
            "Initializing TXTExtractor",
            extra={"file_path": str(self.file_path)}
        )
    
    def extract(self, extract_images: bool = False) -> ExtractionResult[BaseFileMetadata]:
        """
        Extracts content from a text file
        
        Args:
            extract_images: Not used for text files (always False)
        
        Returns:
            ExtractionResult with extracted content and metadata
            (images list will always be empty for text files)
        """
        try:
            self.logger.info(
                "Starting text file extraction",
                extra={
                    "file_path": str(self.file_path),
                    "file_name": self.file_name
                }
            )
            
            # Read text file
            text_content = self._read_text_file()
            
            # Get metadata
            metadata = self.get_metadata()
            
            result = ExtractionResult[BaseFileMetadata](
                content=[text_content],  # List with single element for consistency with PDF
                images=None,  # Text files don't have images
                metadata=metadata
            )
            
            content_length = len(text_content) if text_content else 0
            self.logger.info(
                "Text file extraction completed successfully",
                extra={
                    "file_path": str(self.file_path),
                    "file_name": self.file_name,
                    "content_length": content_length
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"Error extracting content from text file: {str(e)}",
                extra={
                    "file_path": str(self.file_path),
                    "file_name": self.file_name
                },
                exc_info=True
            )
            raise Exception(f"Error extracting content from text file {self.file_name}: {str(e)}")
    
    def _read_text_file(self) -> str:
        """
        Reads the content of the text file
        
        Returns:
            String with the file content
        """
        try:
            self.logger.debug(f"Reading text file: {self.file_name}")
            # Try UTF-8 first (most common encoding)
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.logger.debug(
                "Text file read successfully",
                extra={
                    "file_name": self.file_name,
                    "content_length": len(content)
                }
            )
            return content
            
        except UnicodeDecodeError:
            # If UTF-8 fails, try with error handling (replace problematic characters)
            self.logger.warning(
                f"UTF-8 decode error, trying with error handling: {self.file_name}",
                extra={"file_path": str(self.file_path)}
            )
            try:
                with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                self.logger.debug(
                    "Text file read with error handling",
                    extra={
                        "file_name": self.file_name,
                        "content_length": len(content)
                    }
                )
                return content
            except Exception as e:
                self.logger.error(
                    f"Error reading text file with UTF-8 encoding: {str(e)}",
                    extra={"file_path": str(self.file_path)},
                    exc_info=True
                )
                raise Exception(f"Error reading text file with UTF-8 encoding: {str(e)}")
        except Exception as e:
            self.logger.error(
                f"Error reading text file: {str(e)}",
                extra={"file_path": str(self.file_path)},
                exc_info=True
            )
            raise Exception(f"Error reading text file: {str(e)}")

