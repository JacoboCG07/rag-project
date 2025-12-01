"""
Tests for ChunkingFactory
"""
import pytest
from pathlib import Path
import sys

# Add src to path
# Calculate project root: go up from test file to project root
# test_chunking_factory.py -> chunking/ -> processing/ -> rag/ -> unit_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

from rag.processing.chunking.chunking_factory import ChunkingFactory
from rag.processing.chunking.base_chunker import BaseChunker
from rag.processing.chunking.text_chunker import TextChunker


class TestChunkingFactory:
    """Test class for ChunkingFactory"""
    
    def test_create_chunker_default(self):
        """Test create_chunker with default strategy"""
        chunker = ChunkingFactory.create_chunker()
        
        assert isinstance(chunker, TextChunker)
        assert chunker.chunk_size == 2000
        assert chunker.overlap == 0
        assert chunker.detect_chapters is True
    
    def test_create_chunker_characters_strategy(self):
        """Test create_chunker with 'characters' strategy"""
        chunker = ChunkingFactory.create_chunker(strategy="characters")
        
        assert isinstance(chunker, TextChunker)
    
    def test_create_chunker_custom_parameters(self):
        """Test create_chunker with custom parameters"""
        chunker = ChunkingFactory.create_chunker(
            strategy="characters",
            chunk_size=1000,
            overlap=100,
            detect_chapters=False
        )
        
        assert isinstance(chunker, TextChunker)
        assert chunker.chunk_size == 1000
        assert chunker.overlap == 100
        assert chunker.detect_chapters is False
    
    def test_create_chunker_case_insensitive(self):
        """Test create_chunker with case insensitive strategy name"""
        chunker1 = ChunkingFactory.create_chunker(strategy="CHARACTERS")
        chunker2 = ChunkingFactory.create_chunker(strategy="characters")
        chunker3 = ChunkingFactory.create_chunker(strategy="Characters")
        
        assert isinstance(chunker1, TextChunker)
        assert isinstance(chunker2, TextChunker)
        assert isinstance(chunker3, TextChunker)
    
    def test_create_chunker_invalid_strategy(self):
        """Test create_chunker with invalid strategy"""
        with pytest.raises(ValueError) as exc_info:
            ChunkingFactory.create_chunker(strategy="invalid_strategy")
        
        assert "No chunking strategy found" in str(exc_info.value)
        assert "invalid_strategy" in str(exc_info.value)
    
    def test_register_strategy(self):
        """Test register_strategy method"""
        # Create a mock chunker class for testing
        class MockChunker(BaseChunker):
            def chunk(self, *, texts, return_metadata=False):
                return []
        
        # Register the new strategy
        ChunkingFactory.register_strategy(name="mock", chunker_class=MockChunker)
        
        # Verify it's registered
        chunker = ChunkingFactory.create_chunker(strategy="mock")
        assert isinstance(chunker, MockChunker)
        
        # Clean up: remove the mock strategy
        # Note: This might affect other tests, but for now we'll leave it
        # In a real scenario, you might want to restore the original registry
    
    def test_create_chunker_whitespace_trimming(self):
        """Test that strategy name is trimmed of whitespace"""
        chunker = ChunkingFactory.create_chunker(strategy="  characters  ")
        
        assert isinstance(chunker, TextChunker)
    
    def test_create_chunker_works(self):
        """Test that created chunker actually works"""
        chunker = ChunkingFactory.create_chunker(
            chunk_size=50,
            overlap=10
        )
        
        texts = ["This is a test text that will be chunked."]
        result = chunker.chunk(texts=texts)
        
        assert isinstance(result, list)
        assert len(result) > 0

