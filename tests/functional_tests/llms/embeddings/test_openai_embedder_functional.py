"""
Functional tests for OpenAIEmbedder
These tests make real API calls to OpenAI and require a valid API key.
"""
import pytest
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add src to path
# Calculate project root: go up from test file to project root
# test_openai_embedder_functional.py -> embeddings/ -> llms/ -> functional_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

# Load .env file from project root
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from src.llms.embeddings.openai_embedder import OpenAIEmbedder, OPENAI_AVAILABLE
from src.llms.embeddings.base_embedder import RateLimitError


# Skip all tests if OpenAI package is not available
pytestmark = pytest.mark.skipif(
    not OPENAI_AVAILABLE,
    reason="OpenAI package not installed"
)


def has_openai_api_key():
    """Check if OpenAI API key is available"""
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.fixture
def openai_embedder():
    """Fixture to create OpenAIEmbedder instance if API key is available"""
    if not has_openai_api_key():
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    return OpenAIEmbedder()


@pytest.mark.functional
@pytest.mark.requires_api_key
class TestOpenAIEmbedderFunctional:
    """Functional tests for OpenAIEmbedder that make real API calls"""
    
    def test_generate_embedding_real_api(self, openai_embedder):
        """Test generate_embedding with real OpenAI API call"""
        text = "This is a test text for embedding generation."
        
        embedding, token_count = openai_embedder.generate_embedding(text=text)
        
        # Verify embedding structure
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
        
        # Verify dimensions match expected model dimensions
        expected_dimensions = openai_embedder.dimensions
        assert len(embedding) == expected_dimensions
        
        # Verify token count is provided when count_tokens=True
        assert token_count is not None
        assert isinstance(token_count, int)
        assert token_count > 0
    
    def test_generate_embedding_different_texts(self, openai_embedder):
        """Test that different texts produce different embeddings"""
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "Python is a programming language."
        
        embedding1, _ = openai_embedder.generate_embedding(text=text1)
        embedding2, _ = openai_embedder.generate_embedding(text=text2)
        
        # Embeddings should be different
        assert embedding1 != embedding2
        
        # But should have same dimensions
        assert len(embedding1) == len(embedding2)
    
    def test_generate_embedding_similar_texts(self, openai_embedder):
        """Test that similar texts produce similar embeddings"""
        text1 = "I love programming in Python."
        text2 = "I enjoy coding with Python."
        
        embedding1, _ = openai_embedder.generate_embedding(text=text1)
        embedding2, _ = openai_embedder.generate_embedding(text=text2)
        
        # Calculate cosine similarity (simplified check)
        # Similar texts should have some similarity
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5
        similarity = dot_product / (norm1 * norm2)
        
        # Similarity should be reasonably high (> 0.7) for similar texts
        assert similarity > 0.7, f"Similarity too low: {similarity}"
    
    def test_generate_embedding_empty_text_raises_error(self, openai_embedder):
        """Test that empty text raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            openai_embedder.generate_embedding(text="")
        
        assert "text must be a non-empty string" in str(exc_info.value)
    
    def test_generate_embeddings_batch_real_api(self, openai_embedder):
        """Test generate_embeddings_batch with real API calls"""
        texts = [
            "First text for batch processing.",
            "Second text in the batch.",
            "Third text to test batch functionality."
        ]
        
        results = openai_embedder.generate_embeddings_batch(
            texts=texts,
            batch_size=2
        )
        
        # Verify results structure
        assert isinstance(results, list)
        assert len(results) == len(texts)
        
        # Verify each result
        for result in results:
            assert result is not None
            embedding, token_count = result
            assert isinstance(embedding, list)
            assert len(embedding) == openai_embedder.dimensions
            assert token_count is not None
    
    def test_generate_embeddings_batch_filters_empty_texts(self, openai_embedder):
        """Test that generate_embeddings_batch filters empty texts"""
        texts = [
            "Valid text 1",
            "",
            "   ",
            None,
            "Valid text 2"
        ]
        
        results = openai_embedder.generate_embeddings_batch(texts=texts)
        
        assert len(results) == len(texts)
        # Valid texts should have embeddings
        assert results[0] is not None
        assert results[4] is not None
        # Invalid texts should be None
        assert results[1] is None
        assert results[2] is None
        assert results[3] is None
    
    def test_dimensions_text_embedding_3_small(self):
        """Test dimensions with text-embedding-3-small model"""
        if not has_openai_api_key():
            pytest.skip("OPENAI_API_KEY environment variable not set")
        
        embedder = OpenAIEmbedder(model="text-embedding-3-small")
        dimensions = embedder.dimensions
        
        assert dimensions == 1536
    
    def test_dimensions_text_embedding_3_large(self):
        """Test dimensions with text-embedding-3-large model"""
        if not has_openai_api_key():
            pytest.skip("OPENAI_API_KEY environment variable not set")
        
        embedder = OpenAIEmbedder(model="text-embedding-3-large")
        dimensions = embedder.dimensions
        
        assert dimensions == 3072
    
    def test_dimensions_text_embedding_ada_002(self):
        """Test dimensions with text-embedding-ada-002 model"""
        if not has_openai_api_key():
            pytest.skip("OPENAI_API_KEY environment variable not set")
        
        embedder = OpenAIEmbedder(model="text-embedding-ada-002")
        dimensions = embedder.dimensions
        
        assert dimensions == 1536
    
    def test_generate_embedding_without_token_count(self):
        """Test generate_embedding with count_tokens=False"""
        if not has_openai_api_key():
            pytest.skip("OPENAI_API_KEY environment variable not set")
        
        embedder = OpenAIEmbedder(count_tokens=False)
        text = "Test text without token counting."
        
        embedding, token_count = embedder.generate_embedding(text=text)
        
        assert isinstance(embedding, list)
        assert token_count is None
    
    def test_generate_embedding_long_text(self, openai_embedder):
        """Test generate_embedding with a longer text"""
        # Create a longer text to test token counting
        text = " ".join(["This is a sentence."] * 50)
        
        embedding, token_count = openai_embedder.generate_embedding(text=text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == openai_embedder.dimensions
        assert token_count is not None
        assert token_count > 0
    
    def test_generate_embedding_preserves_semantic_meaning(self, openai_embedder):
        """Test that embeddings preserve semantic meaning"""
        # Synonyms should have similar embeddings
        text1 = "The cat sat on the mat."
        text2 = "The feline rested on the rug."
        
        embedding1, _ = openai_embedder.generate_embedding(text=text1)
        embedding2, _ = openai_embedder.generate_embedding(text=text2)
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5
        similarity = dot_product / (norm1 * norm2)
        
        # Synonyms should have high similarity
        assert similarity > 0.6, f"Semantic similarity too low: {similarity}"

