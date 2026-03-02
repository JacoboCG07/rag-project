"""
Pydantic models for document extraction data structures
"""
from typing import List, Generic, TypeVar, Optional
from pydantic import BaseModel, Field


class ImageData(BaseModel):
    """
    Data structure for a single extracted image
    
    Attributes:
        page: Page number where the image appears (starting at 1)
        image_number_in_page: Image number within that page (starting at 1)
        image_number: Total image number in the document (starting at 1)
        image_base64: Base64 encoded image data (for API usage)
        image_format: Image format (png, jpg, etc.)
    """
    page: int = Field(..., ge=1, description="Page number where the image appears (starting at 1)")
    image_number_in_page: int = Field(..., ge=1, description="Image number within that page (starting at 1)")
    image_number: int = Field(..., ge=1, description="Total image number in the document (starting at 1)")
    image_base64: str = Field(..., description="Base64 encoded image data (for API usage)")
    image_format: str = Field(..., description="Image format (png, jpg, etc.)")


class BaseFileMetadata(BaseModel):
    """
    Base metadata structure for all document types
    
    Attributes:
        file_name: Name of the file
        file_type: File extension/type (e.g., '.pdf', '.docx')
    """
    file_name: str = Field(..., description="Name of the file")
    file_type: str = Field(..., description="File extension/type (e.g., '.pdf', '.docx')")


class PDFFileMetadata(BaseFileMetadata):
    """
    PDF-specific metadata structure
    
    Attributes:
        file_name: Name of the file
        file_type: File extension/type (should be '.pdf')
        total_pages: Total number of pages in the PDF
        total_images: Total number of images in the PDF
    """
    total_pages: int = Field(..., ge=1, description="Total number of pages in the PDF")
    total_images: int = Field(..., ge=0, description="Total number of images in the PDF")
    chapters: bool = Field(..., description="True if the PDF has chapters, False otherwise")


# Type variable for metadata types
MetadataType = TypeVar('MetadataType', bound=BaseFileMetadata)


class ExtractionResult(BaseModel, Generic[MetadataType]):
    """
    Generic result structure for document extraction
    
    Attributes:
        content: List of text content, typically one string per page
        images: Optional list of extracted images (None if no images or extract_images=False)
        metadata: File metadata (specific to document type)
    """
    
    content: List[str] = Field(..., description="List of text content, typically one string per page")
    images: Optional[List[ImageData]] = Field(default=None, description="Optional list of extracted images")
    metadata: MetadataType = Field(..., description="File metadata (specific to document type)")

