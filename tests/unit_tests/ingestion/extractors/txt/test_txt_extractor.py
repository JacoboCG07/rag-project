"""
Tests for TXTExtractor
"""
import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
# Calculate project root: go up from test file to project root
# test_txt_extractor.py -> txt/ -> extractors/ -> ingestion/ -> unit_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

from ingestion.extractors.txt import TXTExtractor

@pytest.fixture
def fixtures_dir():
    """Returns the path to the fixtures directory"""
    return project_root / "tests" / "fixtures"


@pytest.fixture
def sample_text_file():
    """Creates a temporary text file for testing"""
    content = "This is a sample text file.\nIt has multiple lines.\nFor testing purposes."
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def empty_text_file():
    """Creates an empty text file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def fixture_sample_file(fixtures_dir):
    """Returns path to sample.txt fixture file"""
    file_path = fixtures_dir / "sample.txt"
    if not file_path.exists():
        pytest.skip(f"Fixture file not found: {file_path}")
    return str(file_path)




class TestTXTExtractor:
    """Test class for TXTExtractor"""
    
    def test_init(self, sample_text_file):
        """Test TXTExtractor initialization"""
        extractor = TXTExtractor(sample_text_file)
        
        assert extractor.file_path == Path(sample_text_file)
        assert extractor.file_name == os.path.basename(sample_text_file)
        assert extractor.file_type == '.txt'
    
    def test_get_metadata(self, sample_text_file):
        """Test get_metadata method"""
        extractor = TXTExtractor(sample_text_file)
        metadata = extractor.get_metadata()
        
        assert hasattr(metadata, 'file_name')
        assert hasattr(metadata, 'file_type')
        assert metadata.file_type == '.txt'
        assert metadata.file_name == os.path.basename(sample_text_file)
    
    def test_read_text_file(self, sample_text_file):
        """Test _read_text_file method"""
        extractor = TXTExtractor(sample_text_file)
        content = extractor._read_text_file()
        
        assert isinstance(content, str)
        assert "sample text file" in content
        assert "multiple lines" in content
    
    def test_read_text_file_empty(self, empty_text_file):
        """Test reading an empty text file"""
        extractor = TXTExtractor(empty_text_file)
        content = extractor._read_text_file()
        
        assert isinstance(content, str)
        assert content == ""
    
    def test_extract(self, sample_text_file):
        """Test extract method"""
        extractor = TXTExtractor(sample_text_file)
        result = extractor.extract()
        
        assert hasattr(result, 'content')
        assert hasattr(result, 'images')
        assert hasattr(result, 'metadata')
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert isinstance(result.content[0], str)
        assert "sample text file" in result.content[0]
        assert result.images is None  # Text files don't have images
        assert result.metadata.file_type == '.txt'
    
    def test_extract_empty(self, empty_text_file):
        """Test extract with empty file"""
        extractor = TXTExtractor(empty_text_file)
        result = extractor.extract()
        
        assert hasattr(result, 'content')
        assert hasattr(result, 'images')
        assert hasattr(result, 'metadata')
        assert result.content == [""]
        assert result.images is None
    
    def test_extract_fixture(self, fixture_sample_file):
        """Test extract with sample.txt fixture file from tests/fixtures/
        
        Note: The user can modify the content of sample.txt, but the file must be named 'sample.txt'
        """
        extractor = TXTExtractor(fixture_sample_file)
        result = extractor.extract()
        
        # Generic assertions that work with any TXT file content
        assert hasattr(result, 'content')
        assert hasattr(result, 'images')
        assert hasattr(result, 'metadata')
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert isinstance(result.content[0], str)
        assert len(result.content[0]) > 0  # File should have content (not empty)
        assert result.images is None  # Text files don't have images
        
        # Metadata assertions - file must be named 'sample.txt'
        assert result.metadata.file_name == 'sample.txt'
        assert result.metadata.file_type == '.txt'
    
    def test_extract_error(self):
        """Test extract with non-existent file"""
        extractor = TXTExtractor("nonexistent_file.txt")
        
        with pytest.raises(Exception) as exc_info:
            extractor.extract()
        
        assert "Error extracting content" in str(exc_info.value)
    
    def test_read_text_file_error(self):
        """Test _read_text_file with non-existent file"""
        extractor = TXTExtractor("nonexistent_file.txt")
        
        with pytest.raises(Exception) as exc_info:
            extractor._read_text_file()
        
        assert "Error reading text file" in str(exc_info.value)