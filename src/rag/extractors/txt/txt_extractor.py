"""
Text file extractor
"""
from typing import List
from ..base import (
    BaseDocumentExtractor,
    BaseFileMetadata,
    ExtractionResult
)


class TXTExtractor(BaseDocumentExtractor[BaseFileMetadata]):
    """Extractor for text files"""
    
    def __init__(self, file_path: str):
        """
        Initializes the TXT extractor
        
        Args:
            file_path: Path to the text file
        """
        super().__init__(file_path)
    
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
            # Read text file
            text_content = self._read_text_file()
            
            # Get metadata
            metadata = self.get_metadata()
            
            return ExtractionResult[BaseFileMetadata](
                content=[text_content],  # List with single element for consistency with PDF
                images=None,  # Text files don't have images
                metadata=metadata
            )
            
        except Exception as e:
            raise Exception(f"Error extracting content from text file {self.file_name}: {str(e)}")
    
    def _read_text_file(self) -> str:
        """
        Reads the content of the text file
        
        Returns:
            String with the file content
        """
        try:
            # Try UTF-8 first (most common encoding)
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
            
        except UnicodeDecodeError:
            # If UTF-8 fails, try with error handling (replace problematic characters)
            try:
                with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                return content
            except Exception as e:
                raise Exception(f"Error reading text file with UTF-8 encoding: {str(e)}")
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")

