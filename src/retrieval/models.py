"""
Data Transfer Objects (DTOs) for search module
Defines structured data models using Pydantic for type safety and validation
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    """
    Document summary retrieved from Milvus summaries partition.
    Fields match Milvus schema (file_type, pages, chapters, full_images).
    """
    file_id: str = Field(..., description="Unique identifier for the document")
    file_name: str = Field(..., description="Name of the file")
    file_type: str = Field(..., description="Type of file (PDF, TXT, DOCX, etc.)")
    pages: str = Field(default="0", description="Number of pages in the document")
    chapters: str = Field(default="0", description="Chapters info")
    full_images: str = Field(default="0", description="Number of images")
    text: str = Field(..., description="Summary or description of the document")


class DocumentMetadata(BaseModel):
    """
    Metadata extracted from user query for a specific document.
    Used to build precise search filters.
    """
    pages: Optional[List[int]] = Field(None, description="Specific page numbers mentioned in query")
    chapters: Optional[List[str]] = Field(None, description="Specific chapters mentioned in query")
    search_image: bool = Field(False, description="Whether the query is looking for images")
    num_image: Optional[List[int]] = Field(None, description="Specific image numbers mentioned")
    file_type: str = Field(..., description="Type of file to search in")


class DocumentFilter(BaseModel):
    """
    Search filter for a specific document.
    Contains the document ID and its Milvus filter expression.
    """
    id: str = Field(..., description="Document file_id")
    expresion_milvus: str = Field(..., description="Milvus filter expression for this document")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_001",
                "expresion_milvus": 'file_id == "doc_001" and pages in ["1","2","3"]'
            }
        }


class SearchResult(BaseModel):
    """
    Result from a Milvus search operation.
    Contains the matched chunk with its metadata and similarity score.
    """
    id: int = Field(..., description="Chunk ID in Milvus")
    score: float = Field(..., description="Similarity score (0.0 to 1.0)")
    text: str = Field(..., description="Text content of the chunk")
    file_id: str = Field(..., description="Document file_id this chunk belongs to")
    file_name: str = Field(..., description="Name of the source file")
    source_id: str = Field(..., description="Source identifier")
    pages: str = Field(..., description="Page range for this chunk")
    chapters: str = Field(..., description="Chapter information")
    file_type: str = Field(..., description="Type of source file")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": 12345,
                "score": 0.95,
                "text": "Installation instructions...",
                "file_id": "doc_001",
                "file_name": "manual.pdf",
                "source_id": "src_001",
                "pages": "1-5",
                "chapters": "Chapter 1",
                "file_type": "PDF"
            }
        }


class DocumentFilterWithDetails(DocumentFilter):
    """
    Extended document filter with metadata and summary information.
    Used for detailed responses.
    """
    metadata: DocumentMetadata = Field(..., description="Extracted metadata for this document")
    summary: DocumentSummary = Field(..., description="Document summary information")