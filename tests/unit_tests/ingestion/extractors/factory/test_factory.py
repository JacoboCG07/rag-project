"""
Tests for DocumentExtractorFactory
"""
import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
# Calculate project root: go up from test file to project root
# test_factory.py -> factory/ -> extractors/ -> ingestion/ -> unit_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

from ingestion.extractors.factory import DocumentExtractorFactory
from ingestion.extractors.pdf import PDFExtractor
from ingestion.extractors.txt import TXTExtractor
from ingestion.extractors.base import BaseDocumentExtractor  # pyright: ignore[reportMissingImports]

class TestDocumentExtractorFactory:
    """Test class for DocumentExtractorFactory"""
    
    def test_create_extractor_pdf(self):
        """Test creating PDFExtractor for .pdf file"""
        file_path = "test_file.pdf"
        extractor = DocumentExtractorFactory.create_extractor(file_path)
        
        assert isinstance(extractor, PDFExtractor)
        assert isinstance(extractor, BaseDocumentExtractor)
        assert extractor.file_path == Path(file_path)
        assert extractor.file_type == '.pdf'
    
    def test_create_extractor_txt(self):
        """Test creating TXTExtractor for .txt file"""
        file_path = "test_file.txt"
        extractor = DocumentExtractorFactory.create_extractor(file_path)
        
        assert isinstance(extractor, TXTExtractor)
        assert isinstance(extractor, BaseDocumentExtractor)
        assert extractor.file_path == Path(file_path)
        assert extractor.file_type == '.txt'
    
    def test_create_extractor_uppercase_pdf(self):
        """Test creating PDFExtractor for .PDF file (uppercase extension)"""
        file_path = "test_file.PDF"
        extractor = DocumentExtractorFactory.create_extractor(file_path)
        
        assert isinstance(extractor, PDFExtractor)
        assert extractor.file_path == Path(file_path)
        assert extractor.file_type == '.pdf'  # file_type is always lowercase
    
    def test_create_extractor_uppercase_txt(self):
        """Test creating TXTExtractor for .TXT file (uppercase extension)"""
        file_path = "test_file.TXT"
        extractor = DocumentExtractorFactory.create_extractor(file_path)
        
        assert isinstance(extractor, TXTExtractor)
        assert extractor.file_path == Path(file_path)
        assert extractor.file_type == '.txt'  # file_type is always lowercase
    
    def test_create_extractor_mixed_case(self):
        """Test creating extractor for .Pdf file (mixed case extension)"""
        file_path = "test_file.Pdf"
        extractor = DocumentExtractorFactory.create_extractor(file_path)
        
        assert isinstance(extractor, PDFExtractor)
        assert extractor.file_path == Path(file_path)
    
    def test_create_extractor_unsupported_extension(self):
        """Test error when file has unsupported extension"""
        file_path = "test_file.docx"
        
        with pytest.raises(ValueError) as exc_info:
            DocumentExtractorFactory.create_extractor(file_path)
        
        assert "Unsupported file type" in str(exc_info.value)
        assert ".docx" in str(exc_info.value)
    
    def test_create_extractor_no_extension(self):
        """Test error when file has no extension"""
        file_path = "test_file"
        
        with pytest.raises(ValueError) as exc_info:
            DocumentExtractorFactory.create_extractor(file_path)
        
        assert "Unsupported file type" in str(exc_info.value)
    
    def test_create_extractor_path_with_spaces(self):
        """Test creating extractor with file path containing spaces"""
        file_path = "test file with spaces.pdf"
        extractor = DocumentExtractorFactory.create_extractor(file_path)
        
        assert isinstance(extractor, PDFExtractor)
        assert extractor.file_path == Path(file_path)
        assert extractor.file_type == '.pdf'
    
    def test_create_extractor_absolute_path(self):
        """Test creating extractor with absolute path"""
        file_path = "/path/to/file.pdf"
        extractor = DocumentExtractorFactory.create_extractor(file_path)
        
        assert isinstance(extractor, PDFExtractor)
        assert extractor.file_path == Path(file_path)
        assert extractor.file_type == '.pdf'
    
    def test_create_extractor_windows_path(self):
        """Test creating extractor with Windows-style path"""
        file_path = "C:\\Users\\test\\file.txt"
        extractor = DocumentExtractorFactory.create_extractor(file_path)
        
        assert isinstance(extractor, TXTExtractor)
        assert extractor.file_path == Path(file_path)
        assert extractor.file_type == '.txt'