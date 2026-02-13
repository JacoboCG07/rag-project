"""
Tests for DocumentExtractionManager
"""
import pytest
import tempfile
import os
import shutil
from pathlib import Path
import sys
import fitz  # PyMuPDF

# Add src to path
# Calculate project root: go up from test file to project root
# test_folder_reader.py -> folder_reader/ -> ingestion/ -> unit_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

from ingestion.extractors import DocumentExtractionManager
from ingestion.types import ExtractionResult, BaseFileMetadata


@pytest.fixture
def temp_folder_with_files():
    """Creates a temporary folder with PDF and TXT files for testing"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create a PDF file
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Test PDF content")
    pdf_path = Path(temp_dir) / "test_file.pdf"
    doc.save(str(pdf_path))
    doc.close()
    
    # Create a TXT file
    txt_path = Path(temp_dir) / "test_file.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Test TXT content")
    
    # Create an unsupported file (to test filtering)
    unsupported_path = Path(temp_dir) / "test_file.docx"
    with open(unsupported_path, 'w') as f:
        f.write("Unsupported content")
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_empty_folder():
    """Creates an empty temporary folder for testing"""
    temp_dir = tempfile.mkdtemp()
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_folder_only_pdf():
    """Creates a temporary folder with only PDF files"""
    temp_dir = tempfile.mkdtemp()
    
    # Create PDF files
    for i in range(2):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), f"PDF content {i+1}")
        pdf_path = Path(temp_dir) / f"test_file_{i+1}.pdf"
        doc.save(str(pdf_path))
        doc.close()
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_folder_only_txt():
    """Creates a temporary folder with only TXT files"""
    temp_dir = tempfile.mkdtemp()
    
    # Create TXT files
    for i in range(2):
        txt_path = Path(temp_dir) / f"test_file_{i+1}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"TXT content {i+1}")
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestDocumentExtractionManager:
    """Test class for DocumentExtractionManager"""
    
    def test_init(self, temp_folder_with_files):
        """Test DocumentExtractionManager initialization"""
        manager = DocumentExtractionManager(temp_folder_with_files)
        
        assert manager.folder_path == Path(temp_folder_with_files)
        # Supported extensions are automatically obtained from DocumentExtractorFactory
        assert manager.supported_extensions == ['.pdf', '.txt']
    
    def test_get_files(self, temp_folder_with_files):
        """Test get_files method returns supported files"""
        manager = DocumentExtractionManager(temp_folder_with_files)
        files = manager.get_files()
        
        assert isinstance(files, list)
        assert len(files) == 2  # PDF and TXT, not DOCX
        file_extensions = [f.suffix.lower() for f in files]
        assert '.pdf' in file_extensions
        assert '.txt' in file_extensions
        assert '.docx' not in file_extensions
    
    def test_get_files_empty_folder(self, temp_empty_folder):
        """Test get_files with empty folder"""
        manager = DocumentExtractionManager(temp_empty_folder)
        files = manager.get_files()
        
        assert isinstance(files, list)
        assert len(files) == 0
    
    def test_get_files_filters_by_supported_extensions(self, temp_folder_with_files):
        """Test get_files filters files by supported extensions from factory"""
        manager = DocumentExtractionManager(temp_folder_with_files)
        files = manager.get_files()
        
        assert isinstance(files, list)
        # Should return all files with supported extensions (.pdf and .txt)
        assert len(files) >= 1
        # All returned files should have supported extensions
        for file in files:
            assert file.suffix.lower() in manager.supported_extensions
    
    def test_get_files_error_nonexistent_folder(self):
        """Test get_files raises error for non-existent folder"""
        manager = DocumentExtractionManager("nonexistent_folder")
        
        with pytest.raises(Exception) as exc_info:
            manager.get_files()
        
        assert "Error getting files from folder" in str(exc_info.value)
    
    def test_get_files_by_extension_pdf(self, temp_folder_with_files):
        """Test get_files_by_extension for PDF files"""
        manager = DocumentExtractionManager(temp_folder_with_files)
        files = manager.get_files_by_extension('.pdf')
        
        assert isinstance(files, list)
        assert len(files) == 1
        assert files[0].suffix.lower() == '.pdf'
    
    def test_get_files_by_extension_txt(self, temp_folder_with_files):
        """Test get_files_by_extension for TXT files"""
        manager = DocumentExtractionManager(temp_folder_with_files)
        files = manager.get_files_by_extension('.txt')
        
        assert isinstance(files, list)
        assert len(files) == 1
        assert files[0].suffix.lower() == '.txt'
    
    def test_get_files_by_extension_without_dot(self, temp_folder_with_files):
        """Test get_files_by_extension without dot in extension"""
        manager = DocumentExtractionManager(temp_folder_with_files)
        files = manager.get_files_by_extension('pdf')
        
        assert isinstance(files, list)
        assert len(files) == 1
        assert files[0].suffix.lower() == '.pdf'
    
    def test_get_files_by_extension_uppercase(self, temp_folder_with_files):
        """Test get_files_by_extension with uppercase extension"""
        manager = DocumentExtractionManager(temp_folder_with_files)
        files = manager.get_files_by_extension('.PDF')
        
        assert isinstance(files, list)
        assert len(files) == 1
        assert files[0].suffix.lower() == '.pdf'
    
    def test_get_files_by_extension_no_matches(self, temp_folder_with_files):
        """Test get_files_by_extension with extension that has no matches"""
        manager = DocumentExtractionManager(temp_folder_with_files)
        files = manager.get_files_by_extension('.xyz')
        
        assert isinstance(files, list)
        assert len(files) == 0
    
    def test_get_files_by_extension_error_nonexistent_folder(self):
        """Test get_files_by_extension raises error for non-existent folder"""
        manager = DocumentExtractionManager("nonexistent_folder")
        
        with pytest.raises(Exception) as exc_info:
            manager.get_files_by_extension('.pdf')
        
        assert "Error getting files by extension" in str(exc_info.value)
    
    def test_extract_file_data_pdf(self, temp_folder_only_pdf):
        """Test extract_file_data with PDF file"""
        manager = DocumentExtractionManager(temp_folder_only_pdf)
        files = manager.get_files()
        
        result = manager.extract_file_data(files[0], extract_images=False)
        
        assert isinstance(result, ExtractionResult)
        assert hasattr(result, 'content')
        assert hasattr(result, 'images')
        assert hasattr(result, 'metadata')
        assert isinstance(result.content, list)
        assert len(result.content) > 0
    
    def test_extract_file_data_txt(self, temp_folder_only_txt):
        """Test extract_file_data with TXT file"""
        manager = DocumentExtractionManager(temp_folder_only_txt)
        files = manager.get_files()
        
        result = manager.extract_file_data(files[0])
        
        assert isinstance(result, ExtractionResult)
        assert hasattr(result, 'content')
        assert hasattr(result, 'metadata')
        assert isinstance(result.content, list)
        assert len(result.content) == 1
    
    def test_extract_file_data_with_images(self, temp_folder_only_pdf):
        """Test extract_file_data with extract_images=True"""
        manager = DocumentExtractionManager(temp_folder_only_pdf)
        files = manager.get_files()
        
        result = manager.extract_file_data(files[0], extract_images=True)
        
        assert isinstance(result, ExtractionResult)
        assert hasattr(result, 'content')
        assert hasattr(result, 'images')
        assert hasattr(result, 'metadata')
        assert isinstance(result.images, list)
    
    def test_extract_file_data_error_nonexistent_file(self, temp_empty_folder):
        """Test extract_file_data raises error for non-existent file"""
        manager = DocumentExtractionManager(temp_empty_folder)
        nonexistent_file = Path(temp_empty_folder) / "nonexistent.pdf"
        
        with pytest.raises(Exception) as exc_info:
            manager.extract_file_data(nonexistent_file)
        
        assert "Error extracting document" in str(exc_info.value)
    
    def test_extract_files_multiple_files(self, temp_folder_with_files):
        """Test extract_files extracts multiple files"""
        manager = DocumentExtractionManager(temp_folder_with_files)
        results = manager.extract_files(extract_images=False)
        
        assert isinstance(results, list)
        assert len(results) == 2  # PDF and TXT
        
        # Check that each result has the expected structure
        for result in results:
            assert isinstance(result, ExtractionResult)
            assert hasattr(result, 'content')
            assert hasattr(result, 'metadata')
            assert isinstance(result.content, list)
    
    def test_extract_files_empty_folder(self, temp_empty_folder):
        """Test extract_files with empty folder"""
        manager = DocumentExtractionManager(temp_empty_folder)
        results = manager.extract_files()
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_extract_files_with_images(self, temp_folder_only_pdf):
        """Test extract_files with extract_images=True"""
        manager = DocumentExtractionManager(temp_folder_only_pdf)
        results = manager.extract_files(extract_images=True)
        
        assert isinstance(results, list)
        assert len(results) == 2
        
        # Check that images are included in results
        for result in results:
            assert hasattr(result, 'images')
            assert isinstance(result.images, list)
    
    def test_extract_files_max_workers(self, temp_folder_with_files):
        """Test extract_files with custom max_workers"""
        manager = DocumentExtractionManager(temp_folder_with_files)
        results = manager.extract_files(extract_images=False, max_workers=1)
        
        assert isinstance(results, list)
        assert len(results) == 2
    
    def test_extract_files_metadata_in_result(self, temp_folder_with_files):
        """Test that extract_files includes metadata with file_name in each result"""
        manager = DocumentExtractionManager(temp_folder_with_files)
        results = manager.extract_files(extract_images=False)
        
        for result in results:
            assert hasattr(result, 'metadata')
            assert hasattr(result.metadata, 'file_name')
            assert isinstance(result.metadata.file_name, str)
            # Verify file_name is not empty
            assert len(result.metadata.file_name) > 0

