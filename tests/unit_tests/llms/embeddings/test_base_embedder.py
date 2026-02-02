"""
Tests for BaseEmbedder
"""
import pytest
import time
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# Add src to path
# Calculate project root: go up from test file to project root
# test_base_embedder.py -> embeddings/ -> llms/ -> unit_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

from src.llms.embeddings.base_embedder import BaseEmbedder, RateLimitError


class MockEmbedder(BaseEmbedder):
    """Mock implementation of BaseEmbedder for testing"""
    
    def __init__(self, should_raise_rate_limit=False, delay=0, dimensions=128, 
                 always_raise_rate_limit=False, always_raise_error=False, error_message=None):
        self.should_raise_rate_limit = should_raise_rate_limit
        self.delay = delay
        self.call_count = 0
        self.dimensions = dimensions
        self.always_raise_rate_limit = always_raise_rate_limit
        self.always_raise_error = always_raise_error
        self.error_message = error_message or "Connection error"
    
    def generate_embedding(self, text: str):
        """Mock implementation of generate_embedding"""
        self.call_count += 1
        
        if self.delay > 0:
            time.sleep(self.delay)
        
        # Check for always_raise conditions first (for pickle compatibility)
        if self.always_raise_rate_limit:
            raise RateLimitError("Rate limit exceeded")
        
        if self.always_raise_error:
            raise Exception(self.error_message)
        
        if self.should_raise_rate_limit and self.call_count <= 2:
            raise RateLimitError("Rate limit exceeded")
        
        return ([0.1, 0.2, 0.3], 10)
    
    def _get_serializable_config(self) -> dict:
        """Mock implementation of _get_serializable_config"""
        return {
            "dimensions": self.dimensions,
            "should_raise_rate_limit": self.should_raise_rate_limit,
            "always_raise_rate_limit": self.always_raise_rate_limit,
            "always_raise_error": self.always_raise_error,
            "error_message": self.error_message
        }
    
    @classmethod
    def _from_config(cls, config: dict) -> 'MockEmbedder':
        """Mock implementation of _from_config"""
        instance = cls.__new__(cls)
        instance.dimensions = config["dimensions"]
        instance.should_raise_rate_limit = config["should_raise_rate_limit"]
        instance.always_raise_rate_limit = config["always_raise_rate_limit"]
        instance.always_raise_error = config["always_raise_error"]
        instance.error_message = config["error_message"]
        instance.call_count = 0
        instance.delay = 0
        return instance


class TestBaseEmbedder:
    """Test class for BaseEmbedder"""
    
    def test_generate_embeddings_batch_empty(self):
        """Test generate_embeddings_batch with empty list"""
        embedder = MockEmbedder()
        
        result = embedder.generate_embeddings_batch(texts=[])
        
        assert result == []
    
    def test_generate_embeddings_batch_single_text(self):
        """Test generate_embeddings_batch with single text"""
        embedder = MockEmbedder()
        texts = ["Test text"]
        
        result = embedder.generate_embeddings_batch(texts=texts)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == ([0.1, 0.2, 0.3], 10)
    
    def test_generate_embeddings_batch_multiple_texts(self):
        """Test generate_embeddings_batch with multiple texts"""
        embedder = MockEmbedder()
        texts = ["Text 1", "Text 2", "Text 3"]
        
        result = embedder.generate_embeddings_batch(texts=texts)
        
        assert isinstance(result, list)
        assert len(result) == 3
        for r in result:
            assert r == ([0.1, 0.2, 0.3], 10)
    
    def test_generate_embeddings_batch_filters_empty_texts(self):
        """Test generate_embeddings_batch filters empty texts"""
        embedder = MockEmbedder()
        texts = ["Valid text", "", "   ", None, "Another valid text"]
        
        result = embedder.generate_embeddings_batch(texts=texts)
        
        assert isinstance(result, list)
        assert len(result) == len(texts)  # Same length, but None for invalid
        
        # Valid texts should have embeddings
        assert result[0] == ([0.1, 0.2, 0.3], 10)
        assert result[4] == ([0.1, 0.2, 0.3], 10)
        
        # Invalid texts should be None
        assert result[1] is None
        assert result[2] is None
        assert result[3] is None
    
    def test_generate_embeddings_batch_custom_batch_size(self):
        """Test generate_embeddings_batch with custom batch size"""
        embedder = MockEmbedder()
        texts = ["Text " + str(i) for i in range(50)]
        
        result = embedder.generate_embeddings_batch(texts=texts, batch_size=10)
        
        assert isinstance(result, list)
        assert len(result) == 50
    
    def test_generate_embeddings_batch_with_retry(self):
        """Test generate_embeddings_batch with rate limit retry"""
        embedder = MockEmbedder(should_raise_rate_limit=True)
        texts = ["Text 1", "Text 2"]
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = embedder.generate_embeddings_batch(
                texts=texts,
                max_retries=3,
                retry_delay=0.1
            )
        
        assert isinstance(result, list)
        assert len(result) == 2
        # After retries, should succeed
        assert result[0] == ([0.1, 0.2, 0.3], 10)
        assert result[1] == ([0.1, 0.2, 0.3], 10)
    
    def test_generate_embeddings_batch_max_retries_exceeded(self):
        """Test generate_embeddings_batch when max retries exceeded"""
        # Use always_raise_rate_limit flag instead of replacing method (for pickle compatibility)
        embedder = MockEmbedder(always_raise_rate_limit=True)
        
        texts = ["Text 1"]
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            with pytest.raises(Exception) as exc_info:
                embedder.generate_embeddings_batch(
                    texts=texts,
                    max_retries=2,
                    retry_delay=0.1
                )
        
        assert "Failed to generate embedding after" in str(exc_info.value)
        assert "Rate limit exceeded" in str(exc_info.value)
    
    def test_generate_embeddings_batch_other_error(self):
        """Test generate_embeddings_batch with non-rate-limit error"""
        # Use always_raise_error flag instead of replacing method (for pickle compatibility)
        embedder = MockEmbedder(always_raise_error=True, error_message="Connection error")
        
        texts = ["Text 1"]
        
        with pytest.raises(Exception) as exc_info:
            embedder.generate_embeddings_batch(texts=texts)
        
        assert "Error generating embedding" in str(exc_info.value)
    
    def test_process_batch_with_retry_success(self):
        """Test _process_batch_with_retry with successful processing"""
        embedder = MockEmbedder()
        batch = ["Text 1", "Text 2", "Text 3"]
        
        result = BaseEmbedder._process_batch_with_retry(
            batch=batch,
            embedder_class_name="MockEmbedder",
            embedder_module=__name__,
            embedder_config=embedder._get_serializable_config(),
            max_retries=3,
            retry_delay=0.1
        )
        
        assert isinstance(result, list)
        assert len(result) == 3
        for r in result:
            assert r == ([0.1, 0.2, 0.3], 10)
    
    def test_process_batch_with_retry_rate_limit(self):
        """Test _process_batch_with_retry with rate limit and retry"""
        embedder = MockEmbedder(should_raise_rate_limit=True)
        batch = ["Text 1"]
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = BaseEmbedder._process_batch_with_retry(
                batch=batch,
                embedder_class_name="MockEmbedder",
                embedder_module=__name__,
                embedder_config=embedder._get_serializable_config(),
                max_retries=3,
                retry_delay=0.1
            )
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == ([0.1, 0.2, 0.3], 10)
    
    def test_generate_embeddings_batch_preserves_order(self):
        """Test that generate_embeddings_batch preserves order of texts"""
        embedder = MockEmbedder()
        texts = ["First", "Second", "Third", "Fourth", "Fifth"]
        
        result = embedder.generate_embeddings_batch(texts=texts)
        
        assert len(result) == 5
        # All should be valid (not None)
        for r in result:
            assert r is not None
            assert r == ([0.1, 0.2, 0.3], 10)
    
    def test_dimensions_returns_int(self):
        """Test that dimensions is an integer"""
        embedder = MockEmbedder(dimensions=256)
        
        dimensions = embedder.dimensions
        
        assert isinstance(dimensions, int)
        assert dimensions == 256
    
    def test_dimensions_custom_dimensions(self):
        """Test dimensions with custom dimensions"""
        embedder = MockEmbedder(dimensions=512)
        
        dimensions = embedder.dimensions
        
        assert dimensions == 512

