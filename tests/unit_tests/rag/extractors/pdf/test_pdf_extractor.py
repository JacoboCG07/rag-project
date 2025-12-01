"""
Tests for PDFExtractor
"""
import pytest
import tempfile
import os
from pathlib import Path
import sys
import fitz  # PyMuPDF

# Add src to path
# Calculate project root: go up from test file to project root
# test_pdf_extractor.py -> pdf/ -> extractors/ -> rag/ -> unit_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

from rag.extractors.pdf import PDFExtractor


@pytest.fixture
def fixtures_dir():
    """Returns the path to the fixtures directory"""
    return project_root / "tests" / "fixtures"


@pytest.fixture
def sample_pdf_file():
    """Creates a temporary PDF file with text content for testing"""
    # Create a temporary PDF with text
    doc = fitz.open()  # Create a new PDF
    
    # Add first page with text
    page1 = doc.new_page()
    page1.insert_text((50, 50), "This is page 1 of the PDF.")
    page1.insert_text((50, 70), "It contains some sample text.")
    
    # Add second page with text
    page2 = doc.new_page()
    page2.insert_text((50, 50), "This is page 2 of the PDF.")
    page2.insert_text((50, 70), "More sample text here.")
    
    # Save to temporary file
    temp_path = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False).name
    doc.save(temp_path)
    doc.close()
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def empty_pdf_file():
    """Creates an empty PDF file (PDF with no content) for testing"""
    # Create a PDF with one empty page
    doc = fitz.open()  # Create a new PDF
    doc.new_page()  # Add one empty page
    
    # Save to temporary file
    temp_path = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False).name
    doc.save(temp_path)
    doc.close()
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def fixture_sample_pdf(fixtures_dir):
    """Returns path to sample.pdf fixture file from tests/fixtures/
    
    Note: The user can modify the content of sample.pdf, but the file must be named 'sample.pdf'
    """
    file_path = fixtures_dir / "sample.pdf"
    if not file_path.exists():
        pytest.skip(f"Fixture file not found: {file_path}")
    return str(file_path)


@pytest.fixture
def fixture_sample_with_images_pdf(fixtures_dir):
    """Returns path to sample_with_images.pdf fixture file from tests/fixtures/
    
    Note: The user can modify the content of sample_with_images.pdf, but the file must be named 'sample_with_images.pdf'
    """
    file_path = fixtures_dir / "sample_with_images.pdf"
    if not file_path.exists():
        pytest.skip(f"Fixture file not found: {file_path}")
    return str(file_path)


class TestPDFExtractor:
    """Test class for PDFExtractor"""
    
    def test_init(self, sample_pdf_file):
        """Test PDFExtractor initialization"""
        extractor = PDFExtractor(sample_pdf_file)
        
        assert extractor.file_path == Path(sample_pdf_file)
        assert extractor.file_name == os.path.basename(sample_pdf_file)
        assert extractor.file_type == '.pdf'
    
    def test_get_metadata(self, sample_pdf_file):
        """Test get_metadata method"""
        extractor = PDFExtractor(sample_pdf_file)
        metadata = extractor.get_metadata()
        
        assert hasattr(metadata, 'file_name')
        assert hasattr(metadata, 'file_type')
        assert hasattr(metadata, 'total_pages')
        assert hasattr(metadata, 'total_images')
        assert metadata.file_type == '.pdf'
        assert metadata.file_name == os.path.basename(sample_pdf_file)
        assert metadata.total_pages >= 1
        assert metadata.total_images >= 0
    
    def test_extract_text_from_pdf(self, sample_pdf_file):
        """Test _extract_text_from_pdf method"""
        extractor = PDFExtractor(sample_pdf_file)
        text_pages = extractor._extract_text_from_pdf()
        
        assert isinstance(text_pages, list)
        assert len(text_pages) == 2  # Two pages in the test PDF
        assert isinstance(text_pages[0], str)
        assert isinstance(text_pages[1], str)
        assert "page 1" in text_pages[0].lower()
        assert "page 2" in text_pages[1].lower()
    
    def test_extract_text_from_pdf_empty(self, empty_pdf_file):
        """Test _extract_text_from_pdf with empty PDF"""
        extractor = PDFExtractor(empty_pdf_file)
        text_pages = extractor._extract_text_from_pdf()
        
        assert isinstance(text_pages, list)
        assert len(text_pages) == 1  # One empty page
        assert isinstance(text_pages[0], str)
        # Empty page may return empty string or whitespace
        assert text_pages[0].strip() == ""
    
    def test_extract(self, sample_pdf_file):
        """Test extract method without images"""
        extractor = PDFExtractor(sample_pdf_file)
        result = extractor.extract(extract_images=False)
        
        assert hasattr(result, 'content')
        assert hasattr(result, 'images')
        assert hasattr(result, 'metadata')
        assert isinstance(result.content, list)
        assert len(result.content) == 2
        assert result.images is None  # None when extract_images=False
        assert result.metadata.file_type == '.pdf'
    
    def test_extract_empty(self, empty_pdf_file):
        """Test extract with empty PDF"""
        extractor = PDFExtractor(empty_pdf_file)
        result = extractor.extract(extract_images=False)
        
        assert hasattr(result, 'content')
        assert hasattr(result, 'images')
        assert hasattr(result, 'metadata')
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert result.images is None
        assert result.content[0].strip() == ""
    
    def test_extract_with_images_no_images_in_pdf(self, sample_pdf_file):
        """Test extract with extract_images=True but PDF has no images"""
        extractor = PDFExtractor(sample_pdf_file)
        result = extractor.extract(extract_images=True)
        
        assert hasattr(result, 'content')
        assert hasattr(result, 'images')
        assert hasattr(result, 'metadata')
        assert isinstance(result.content, list)
        assert len(result.content) == 2
        assert isinstance(result.images, list)  # Empty list when PDF has no images
        assert len(result.images) == 0  # Should be empty list, not None
        assert result.metadata.file_type == '.pdf'
    
    def test_extract_fixture(self, fixture_sample_pdf):
        """Test extract with sample.pdf fixture file from tests/fixtures/
        
        Note: The user can modify the content of sample.pdf, but the file must be named 'sample.pdf'
        """
        extractor = PDFExtractor(fixture_sample_pdf)
        result = extractor.extract(extract_images=False)
        
        # Generic assertions that work with any PDF file content
        assert hasattr(result, 'content')
        assert hasattr(result, 'images')
        assert hasattr(result, 'metadata')
        assert isinstance(result.content, list)
        assert len(result.content) > 0  # At least one page
        assert result.images is None  # None when extract_images=False
        
        # Metadata assertions - file must be named 'sample.pdf'
        assert result.metadata.file_name == 'sample.pdf'
        assert result.metadata.file_type == '.pdf'
    
    def test_extract_fixture_with_images(self, fixture_sample_with_images_pdf):
        """Test extract with sample_with_images.pdf fixture file from tests/fixtures/
        
        Note: The user can modify the content of sample_with_images.pdf, but the file must be named 'sample_with_images.pdf'
        """
        extractor = PDFExtractor(fixture_sample_with_images_pdf)
        result = extractor.extract(extract_images=True)
        
        # Generic assertions that work with any PDF file content
        assert hasattr(result, 'content')
        assert hasattr(result, 'images')
        assert hasattr(result, 'metadata')
        assert isinstance(result.content, list)
        assert len(result.content) > 0  # At least one page
        assert result.images is not None
        assert isinstance(result.images, list)
        assert len(result.images) > 0  # Should have images when extract_images=True
        
        # Metadata assertions - file must be named 'sample_with_images.pdf'
        assert result.metadata.file_name == 'sample_with_images.pdf'
        assert result.metadata.file_type == '.pdf'
    
    def test_extract_images_from_pdf(self, fixture_sample_with_images_pdf):
        """Test _extract_images_from_pdf method with PDF that has images"""
        extractor = PDFExtractor(fixture_sample_with_images_pdf)
        images = extractor._extract_images_from_pdf()
        
        assert isinstance(images, list)
        assert len(images) > 0  # Should have at least one image
        
        # Verify structure of each image (ImageData Pydantic model)
        for image in images:
            assert hasattr(image, 'page')
            assert hasattr(image, 'image_number_in_page')
            assert hasattr(image, 'image_number')
            assert hasattr(image, 'image_base64')
            assert hasattr(image, 'image_format')
            
            # Verify types
            assert isinstance(image.page, int)
            assert isinstance(image.image_number_in_page, int)
            assert isinstance(image.image_number, int)
            assert isinstance(image.image_base64, str)
            assert isinstance(image.image_format, str)
            
            # Verify values are valid
            assert image.page >= 1
            assert image.image_number_in_page >= 1
            assert image.image_number >= 1
            assert len(image.image_base64) > 0  # Base64 string should not be empty
            assert len(image.image_format) > 0  # Format should not be empty
            
            # Verify base64 is valid - try to decode it
            # If it can be decoded, it's valid base64
            import base64
            decoded = base64.b64decode(image.image_base64, validate=True)
            assert len(decoded) > 0  # Decoded image data should not be empty
    
    def test_extract_images_from_pdf_image_numbering(self, fixture_sample_with_images_pdf):
        """Test that image numbering is correct (page, image_number_in_page, total_image_number)"""
        extractor = PDFExtractor(fixture_sample_with_images_pdf)
        images = extractor._extract_images_from_pdf()
        
        if len(images) == 0:
            pytest.skip("PDF has no images to test numbering")
        
        # Verify image_number is sequential starting from 1
        for i, image in enumerate(images):
            assert image.image_number == i + 1, f"Image {i} should have image_number={i+1}, got {image.image_number}"
        
        # Verify page numbers are valid (should be >= 1)
        pages = set(image.page for image in images)
        assert all(page >= 1 for page in pages)
        
        # Verify image_number_in_page for each page
        from collections import defaultdict
        images_by_page = defaultdict(list)
        for image in images:
            images_by_page[image.page].append(image)
        
        for page_num, page_images in images_by_page.items():
            # Images on the same page should have sequential image_number_in_page
            page_images.sort(key=lambda x: x.image_number_in_page)
            for i, image in enumerate(page_images):
                assert image.image_number_in_page == i + 1, f"Page {page_num}, image {i} should have image_number_in_page={i+1}, got {image.image_number_in_page}"
    
    def test_extract_error(self):
        """Test extract with non-existent file"""
        extractor = PDFExtractor("nonexistent_file.pdf")
        
        with pytest.raises(Exception) as exc_info:
            extractor.extract()
        
        assert "Error extracting content" in str(exc_info.value)
    
    def test_extract_text_from_pdf_error(self):
        """Test _extract_text_from_pdf with non-existent file"""
        extractor = PDFExtractor("nonexistent_file.pdf")
        
        with pytest.raises(Exception) as exc_info:
            extractor._extract_text_from_pdf()
        
        assert "Error extracting text from PDF" in str(exc_info.value)

