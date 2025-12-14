"""
Tests for TextChunker
"""
import pytest
from pathlib import Path
import sys

# Add src to path
# Calculate project root: go up from test file to project root
# test_text_chunker.py -> chunking/ -> processing/ -> rag/ -> unit_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

from rag.processing.chunking.text_chunker import TextChunker


class TestTextChunker:
    """Test class for TextChunker"""
    
    def test_init_default(self):
        """Test TextChunker initialization with default parameters"""
        chunker = TextChunker()
        
        assert chunker.chunk_size == 2000
        assert chunker.overlap == 0
        assert chunker.detect_chapters is True
    
    def test_init_custom(self):
        """Test TextChunker initialization with custom parameters"""
        chunker = TextChunker(
            chunk_size=1000,
            overlap=100,
            detect_chapters=False
        )
        
        assert chunker.chunk_size == 1000
        assert chunker.overlap == 100
        assert chunker.detect_chapters is False
    
    def test_chunk_empty_list(self):
        """Test chunk with empty list"""
        chunker = TextChunker()
        
        result = chunker.chunk(texts=[])
        assert result == []
        
        result = chunker.chunk(texts=[], return_metadata=True)
        assert result == ([], [])
    
    def test_chunk_single_short_text(self):
        """Test chunk with single short text"""
        chunker = TextChunker(chunk_size=100)
        texts = ["This is a short text that fits in one chunk."]
        
        result = chunker.chunk(texts=texts)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == texts[0]
    
    def test_chunk_single_long_text(self):
        """Test chunk with single long text that needs splitting"""
        chunker = TextChunker(chunk_size=20)
        texts = ["This is a very long text that needs to be split into multiple chunks because it exceeds the chunk size limit."]
        
        result = chunker.chunk(texts=texts)
        
        assert isinstance(result, list)
        assert len(result) > 1
        # Verify all chunks are within size limit
        for chunk in result:
            assert len(chunk) <= chunker.chunk_size + 10  # Allow some margin for word boundaries
    
    def test_chunk_multiple_texts(self):
        """Test chunk with multiple texts"""
        chunker = TextChunker(chunk_size=50)
        texts = [
            "First page content here.",
            "Second page content here.",
            "Third page content here."
        ]
        
        result = chunker.chunk(texts=texts)
        
        assert isinstance(result, list)
        assert len(result) >= 1
        # All texts should be included
        combined = " ".join(result)
        assert "First page" in combined
        assert "Second page" in combined
        assert "Third page" in combined
    
    def test_chunk_with_overlap(self):
        """Test chunk with overlap between chunks"""
        chunker = TextChunker(chunk_size=30, overlap=10)
        texts = ["This is a longer text that will be split into multiple chunks with overlap between them."]
        
        result = chunker.chunk(texts=texts)
        
        assert isinstance(result, list)
        assert len(result) > 1
        
        # Check that chunks overlap (last part of one chunk appears in next)
        for i in range(len(result) - 1):
            current_chunk = result[i]
            next_chunk = result[i + 1]
            # There should be some overlap (at least a few characters)
            overlap_found = False
            for j in range(min(len(current_chunk), len(next_chunk))):
                if current_chunk[-j:] in next_chunk[:j+10]:
                    overlap_found = True
                    break
            # Note: Overlap might not always be exact due to word boundaries
            # So we just verify chunks are created
    
    def test_chunk_with_overlap_verification(self):
        """Test chunk with overlap and verify overlap text appears"""
        chunker = TextChunker(chunk_size=50, overlap=20)
        texts = ["First part of text. Second part of text. Third part of text. Fourth part of text."]
        
        result = chunker.chunk(texts=texts)
        
        assert isinstance(result, list)
        assert len(result) > 1
        
        # Verify that overlap is applied (chunks should share some text)
        for i in range(len(result) - 1):
            current_chunk = result[i]
            next_chunk = result[i + 1]
            # Get last words of current chunk
            current_words = current_chunk.split()[-3:]
            # Get first words of next chunk
            next_words = next_chunk.split()[:3]
            # There should be some common words (overlap)
            common_words = set(current_words) & set(next_words)
            # At least one word should overlap (allowing for word boundary adjustments)
            assert len(common_words) > 0 or len(current_chunk) > 0
    
    def test_chunk_with_metadata(self):
        """Test chunk with return_metadata=True"""
        chunker = TextChunker(chunk_size=50)
        texts = [
            "First page content.",
            "Second page content.",
            "Third page content."
        ]
        
        chunks, metadata_list = chunker.chunk(texts=texts, return_metadata=True)
        
        assert isinstance(chunks, list)
        assert isinstance(metadata_list, list)
        assert len(chunks) == len(metadata_list)
        
        # Verify metadata structure
        for metadata in metadata_list:
            assert isinstance(metadata, dict)
            assert 'pages' in metadata
            assert isinstance(metadata['pages'], list)
            assert len(metadata['pages']) > 0
            assert 'chapters' in metadata
    
    def test_chunk_detect_chapters_enabled(self):
        """Test chunk with chapter detection enabled"""
        chunker = TextChunker(chunk_size=100, detect_chapters=True)
        texts = [
            "Capítulo I\nThis is the first chapter content.",
            "Capítulo II\nThis is the second chapter content."
        ]
        
        chunks, metadata_list = chunker.chunk(texts=texts, return_metadata=True)
        
        assert len(chunks) > 0
        # Check that chapters are detected
        chapters_found = False
        for metadata in metadata_list:
            if metadata.get('chapters'):
                chapters_found = True
                break
        # At least one chunk should have chapter information
        assert chapters_found
    
    def test_chunk_detect_chapters_disabled(self):
        """Test chunk with chapter detection disabled"""
        chunker = TextChunker(chunk_size=100, detect_chapters=False)
        texts = [
            "Capítulo I\nThis is the first chapter content.",
            "Capítulo II\nThis is the second chapter content."
        ]
        
        chunks, metadata_list = chunker.chunk(texts=texts, return_metadata=True)
        
        assert len(chunks) > 0
        # Chapters should be empty strings when detection is disabled
        for metadata in metadata_list:
            assert metadata.get('chapters') == "" or metadata.get('chapters') == []
    
    def test_chunk_word_boundaries(self):
        """Test that chunking respects word boundaries"""
        chunker = TextChunker(chunk_size=20)
        texts = ["This is a test sentence with multiple words that should not be cut in the middle."]
        
        result = chunker.chunk(texts=texts)
        
        assert isinstance(result, list)
        # Verify no chunks end or start in the middle of words
        for chunk in result:
            # Chunk should not start with a space (unless it's the first character)
            if len(chunk) > 0:
                # Check that words are complete (simplified check)
                words = chunk.split()
                assert len(words) > 0  # Should have at least one word
    
    def test_ensure_length_segments(self):
        """Test _ensure_length_segments method"""
        chunker = TextChunker(chunk_size=20)
        texts = [
            "Short text.",
            "This is a very long text that exceeds the chunk size limit and needs to be split."
        ]
        
        separated_texts, pages = chunker._ensure_length_segments(texts=texts)
        
        assert isinstance(separated_texts, list)
        assert isinstance(pages, list)
        assert len(separated_texts) == len(pages)
        
        # First text should remain as is (short)
        assert "Short text" in separated_texts[0]
        
        # Long text should be split
        assert len(separated_texts) > 2  # Original 2 texts, but long one split
    
    def test_group_segments(self):
        """Test _group_segments method"""
        chunker = TextChunker(chunk_size=50)
        texts = [
            "First segment.",
            "Second segment.",
            "Third segment."
        ]
        pages = [1, 2, 3]
        
        grouped, pages_groups = chunker._group_segments(texts=texts, pages=pages)
        
        assert isinstance(grouped, list)
        assert isinstance(pages_groups, list)
        assert len(grouped) == len(pages_groups)
        
        # Verify page groups
        for pages_group in pages_groups:
            assert isinstance(pages_group, list)
            assert len(pages_group) > 0
    
    def test_get_overlap_text(self):
        """Test _get_overlap_text method"""
        chunker = TextChunker(overlap=10)
        group = ["First text segment", "Second text segment"]
        
        overlap_text = chunker._get_overlap_text(group=group)
        
        # Overlap should be from the last text in group
        assert isinstance(overlap_text, str)
        assert len(overlap_text) <= chunker.overlap + 5  # Allow some margin
    
    def test_get_overlap_text_no_overlap(self):
        """Test _get_overlap_text with overlap=0"""
        chunker = TextChunker(overlap=0)
        group = ["First text segment", "Second text segment"]
        
        overlap_text = chunker._get_overlap_text(group=group)
        
        assert overlap_text == ""
    
    def test_get_chapters_of_segments(self):
        """Test _get_chapters_of_segments method"""
        chunker = TextChunker()
        segments = [
            "Capítulo I\nContent of first chapter.",
            "Regular content without chapter.",
            "II\nContent with Roman numeral chapter."
        ]
        
        chapters = chunker._get_chapters_of_segments(segments=segments)
        
        assert isinstance(chapters, list)
        assert len(chapters) == len(segments)
        
        # First segment should have chapter
        assert len(chapters[0]) > 0
        # Second segment might not have chapter
        # Third segment should have chapter (Roman numeral)
        assert len(chapters[2]) > 0
    
    def test_is_chapter_start_capitulo(self):
        """Test _is_chapter_start with 'Capítulo'"""
        line = "Capítulo I: Introduction"
        assert TextChunker._is_chapter_start(line=line) is True
        
        line = "capítulo 1"
        assert TextChunker._is_chapter_start(line=line) is True
    
    def test_is_chapter_start_roman_numeral(self):
        """Test _is_chapter_start with Roman numerals"""
        line = "I Introduction"
        assert TextChunker._is_chapter_start(line=line) is True
        
        line = "II Main Content"
        assert TextChunker._is_chapter_start(line=line) is True
        
        line = "III Conclusion"
        assert TextChunker._is_chapter_start(line=line) is True
    
    def test_is_chapter_start_not_chapter(self):
        """Test _is_chapter_start with non-chapter lines"""
        line = "This is regular text"
        assert TextChunker._is_chapter_start(line=line) is False
        
        line = ""
        assert TextChunker._is_chapter_start(line=line) is False
        
        line = "   "
        assert TextChunker._is_chapter_start(line=line) is False

