"""
Tests for OpenAIEmbedder
"""
import pytest
import os
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src to path
# Calculate project root: go up from test file to project root
# test_openai_embedder.py -> embeddings/ -> llms/ -> unit_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

from src.llms.embeddings.openai_embedder import OpenAIEmbedder, OPENAI_AVAILABLE
from src.llms.embeddings.base_embedder import RateLimitError


@pytest.fixture
def mock_openai_client():
    """Creates a mock OpenAI client"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.data = [Mock()]
    mock_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_response.usage = Mock()
    mock_response.usage.total_tokens = 10
    mock_client.embeddings.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_api_key():
    """Returns a mock API key"""
    return "test-api-key-12345"


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI package not installed")
class TestOpenAIEmbedder:
    """Test class for OpenAIEmbedder"""
    
    def test_init_with_api_key(self, mock_api_key):
        """Test OpenAIEmbedder initialization with API key"""
        with patch('src.llms.embeddings.openai_embedder.OpenAI') as mock_openai:
            embedder = OpenAIEmbedder(api_key=mock_api_key)
            
            assert embedder.api_key == mock_api_key
            assert embedder.model == "text-embedding-3-small"
            assert embedder.count_tokens is True
            mock_openai.assert_called_once_with(api_key=mock_api_key)
    
    def test_init_with_env_var(self, mock_api_key):
        """Test OpenAIEmbedder initialization with environment variable"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': mock_api_key}):
            with patch('src.llms.embeddings.openai_embedder.OpenAI') as mock_openai:
                embedder = OpenAIEmbedder()
                
                assert embedder.api_key == mock_api_key
                mock_openai.assert_called_once_with(api_key=mock_api_key)
    
    def test_init_no_api_key(self):
        """Test OpenAIEmbedder initialization without API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                OpenAIEmbedder()
            
            assert "OpenAI API key is required" in str(exc_info.value)
    
    def test_init_custom_model(self, mock_api_key):
        """Test OpenAIEmbedder initialization with custom model"""
        with patch('src.llms.embeddings.openai_embedder.OpenAI'):
            embedder = OpenAIEmbedder(
                api_key=mock_api_key,
                model="text-embedding-3-large",
                count_tokens=False
            )
            
            assert embedder.model == "text-embedding-3-large"
            assert embedder.count_tokens is False
    
    def test_generate_embedding_success(self, mock_api_key, mock_openai_client):
        """Test generate_embedding with successful response"""
        with patch('src.llms.embeddings.openai_embedder.OpenAI', return_value=mock_openai_client):
            embedder = OpenAIEmbedder(api_key=mock_api_key)
            
            embedding, token_count = embedder.generate_embedding(text="Test text")
            
            assert isinstance(embedding, list)
            assert len(embedding) == 5
            assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
            assert token_count == 10
            mock_openai_client.embeddings.create.assert_called_once()
    
    def test_generate_embedding_without_token_count(self, mock_api_key):
        """Test generate_embedding with count_tokens=False"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_response
        
        with patch('src.llms.embeddings.openai_embedder.OpenAI', return_value=mock_client):
            embedder = OpenAIEmbedder(api_key=mock_api_key, count_tokens=False)
            
            embedding, token_count = embedder.generate_embedding(text="Test text")
            
            assert isinstance(embedding, list)
            assert token_count is None
    
    def test_generate_embedding_fallback_token_count(self, mock_api_key):
        """Test generate_embedding with fallback token count estimation"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        # Simulate no usage attribute
        del mock_response.usage
        mock_client.embeddings.create.return_value = mock_response
        
        with patch('src.llms.embeddings.openai_embedder.OpenAI', return_value=mock_client):
            embedder = OpenAIEmbedder(api_key=mock_api_key, count_tokens=True)
            
            text = "Test text with approximately 40 characters"
            embedding, token_count = embedder.generate_embedding(text=text)
            
            assert isinstance(embedding, list)
            # Token count should be estimated (roughly len(text) // 4)
            assert token_count is not None
            assert token_count > 0
    
    def test_generate_embedding_empty_text(self, mock_api_key):
        """Test generate_embedding with empty text"""
        with patch('src.llms.embeddings.openai_embedder.OpenAI'):
            embedder = OpenAIEmbedder(api_key=mock_api_key)
            
            with pytest.raises(ValueError) as exc_info:
                embedder.generate_embedding(text="")
            
            assert "text must be a non-empty string" in str(exc_info.value)
    
    def test_generate_embedding_invalid_text(self, mock_api_key):
        """Test generate_embedding with invalid text type"""
        with patch('src.llms.embeddings.openai_embedder.OpenAI'):
            embedder = OpenAIEmbedder(api_key=mock_api_key)
            
            with pytest.raises(ValueError) as exc_info:
                embedder.generate_embedding(text=None)
            
            assert "text must be a non-empty string" in str(exc_info.value)
    
    def test_generate_embedding_rate_limit_error(self, mock_api_key):
        """Test generate_embedding with rate limit error"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("429 Too Many Requests")
        
        with patch('src.llms.embeddings.openai_embedder.OpenAI', return_value=mock_client):
            embedder = OpenAIEmbedder(api_key=mock_api_key)
            
            with pytest.raises(RateLimitError) as exc_info:
                embedder.generate_embedding(text="Test text")
            
            assert "Rate limit exceeded" in str(exc_info.value)
    
    def test_generate_embedding_rate_limit_error_variations(self, mock_api_key):
        """Test generate_embedding with different rate limit error messages"""
        error_messages = [
            "429",
            "Too Many Requests",
            "rate_limit exceeded"
        ]
        
        for error_msg in error_messages:
            mock_client = Mock()
            mock_client.embeddings.create.side_effect = Exception(error_msg)
            
            with patch('src.llms.embeddings.openai_embedder.OpenAI', return_value=mock_client):
                embedder = OpenAIEmbedder(api_key=mock_api_key)
                
                with pytest.raises(RateLimitError):
                    embedder.generate_embedding(text="Test text")
    
    def test_generate_embedding_general_error(self, mock_api_key):
        """Test generate_embedding with general error"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("Connection error")
        
        with patch('src.llms.embeddings.openai_embedder.OpenAI', return_value=mock_client):
            embedder = OpenAIEmbedder(api_key=mock_api_key)
            
            with pytest.raises(Exception) as exc_info:
                embedder.generate_embedding(text="Test text")
            
            assert "Error generating embedding with OpenAI" in str(exc_info.value)
    
    def test_generate_embedding_strips_text(self, mock_api_key, mock_openai_client):
        """Test that generate_embedding strips whitespace from text"""
        with patch('src.llms.embeddings.openai_embedder.OpenAI', return_value=mock_openai_client):
            embedder = OpenAIEmbedder(api_key=mock_api_key)
            
            embedder.generate_embedding(text="  Test text  ")
            
            # Verify that create was called with stripped text
            call_args = mock_openai_client.embeddings.create.call_args
            assert call_args[1]['input'] == "Test text"
    
    def test_dimensions_default_model(self, mock_api_key):
        """Test dimensions with default model (text-embedding-3-small)"""
        with patch('src.llms.embeddings.openai_embedder.OpenAI'):
            embedder = OpenAIEmbedder(api_key=mock_api_key)
            
            dimensions = embedder.dimensions
            
            assert dimensions == 1536
    
    def test_dimensions_text_embedding_3_small(self, mock_api_key):
        """Test dimensions with text-embedding-3-small model"""
        with patch('src.llms.embeddings.openai_embedder.OpenAI'):
            embedder = OpenAIEmbedder(
                api_key=mock_api_key,
                model="text-embedding-3-small"
            )
            
            dimensions = embedder.dimensions
            
            assert dimensions == 1536
    
    def test_dimensions_text_embedding_3_large(self, mock_api_key):
        """Test dimensions with text-embedding-3-large model"""
        with patch('src.llms.embeddings.openai_embedder.OpenAI'):
            embedder = OpenAIEmbedder(
                api_key=mock_api_key,
                model="text-embedding-3-large"
            )
            
            dimensions = embedder.dimensions
            
            assert dimensions == 3072
    
    def test_dimensions_text_embedding_ada_002(self, mock_api_key):
        """Test dimensions with text-embedding-ada-002 model"""
        with patch('src.llms.embeddings.openai_embedder.OpenAI'):
            embedder = OpenAIEmbedder(
                api_key=mock_api_key,
                model="text-embedding-ada-002"
            )
            
            dimensions = embedder.dimensions
            
            assert dimensions == 1536
    
    def test_dimensions_text_embedding_2(self, mock_api_key):
        """Test dimensions with text-embedding-2 model"""
        with patch('src.llms.embeddings.openai_embedder.OpenAI'):
            embedder = OpenAIEmbedder(
                api_key=mock_api_key,
                model="text-embedding-2"
            )
            
            dimensions = embedder.dimensions
            
            assert dimensions == 1536
    
    def test_dimensions_unknown_model(self, mock_api_key):
        """Test dimensions with unknown model raises ValueError"""
        with patch('src.llms.embeddings.openai_embedder.OpenAI'):
            with pytest.raises(ValueError) as exc_info:
                embedder = OpenAIEmbedder(
                    api_key=mock_api_key,
                    model="unknown-model"
                )
            
            assert "Unknown model" in str(exc_info.value)
            assert "unknown-model" in str(exc_info.value)
            assert "MODEL_DIMENSIONS" in str(exc_info.value)

