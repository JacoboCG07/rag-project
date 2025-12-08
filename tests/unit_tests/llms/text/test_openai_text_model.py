"""
Tests for OpenAITextModel
"""
import pytest
import os
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src to path
# Calculate project root: go up from test file to project root
# test_openai_text_model.py -> text/ -> llms/ -> unit_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

from llms.text.openai_text_model import OpenAITextModel, OPENAI_AVAILABLE


@pytest.fixture
def mock_openai_client():
    """Creates a mock OpenAI client"""
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "Test response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_api_key():
    """Returns a mock API key"""
    return "test-api-key-12345"


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI package not installed")
class TestOpenAITextModel:
    """Test class for OpenAITextModel"""
    
    def test_init_with_env_var(self, mock_api_key):
        """Test OpenAITextModel initialization with environment variable"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': mock_api_key}):
            with patch('llms.text.openai_text_model.OpenAI') as mock_openai:
                model = OpenAITextModel()
                
                assert model.api_key == mock_api_key
                assert model.model == "gpt-4o"
                assert model.default_max_tokens == 10_000
                assert model.default_temperature == 0.3
                mock_openai.assert_called_once_with(api_key=mock_api_key)
    
    
    def test_init_no_api_key(self):
        """Test OpenAITextModel initialization without API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                OpenAITextModel()
            
            assert "OpenAI API key is required" in str(exc_info.value)
    
    def test_init_custom_parameters(self, mock_api_key):
        """Test OpenAITextModel initialization with custom parameters"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': mock_api_key}):
            with patch('llms.text.openai_text_model.OpenAI'):
                model = OpenAITextModel(
                    model="gpt-3.5-turbo",
                    max_tokens=2000,
                    temperature=0.5
                )
                
                assert model.model == "gpt-3.5-turbo"
                assert model.default_max_tokens == 2000
                assert model.default_temperature == 0.5
    
    def test_call_text_model_with_prompt(self, mock_api_key, mock_openai_client):
        """Test call_text_model with simple prompt"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': mock_api_key}):
            with patch('llms.text.openai_text_model.OpenAI', return_value=mock_openai_client):
                model = OpenAITextModel()
            
            result = model.call_text_model(prompt="Test prompt")
            
            assert result == "Test response"
            mock_openai_client.chat.completions.create.assert_called_once()
            
            # Verify call arguments
            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]['model'] == "gpt-4o"
            assert len(call_args[1]['messages']) == 1
            assert call_args[1]['messages'][0]['role'] == "user"
            assert call_args[1]['messages'][0]['content'] == "Test prompt"
    
    def test_call_text_model_with_system_prompt(self, mock_api_key, mock_openai_client):
        """Test call_text_model with system prompt"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': mock_api_key}):
            with patch('llms.text.openai_text_model.OpenAI', return_value=mock_openai_client):
                model = OpenAITextModel()
            
            result = model.call_text_model(
                prompt="User prompt",
                system_prompt="You are a helpful assistant"
            )
            
            assert result == "Test response"
            call_args = mock_openai_client.chat.completions.create.call_args
            messages = call_args[1]['messages']
            assert len(messages) == 2
            assert messages[0]['role'] == "system"
            assert messages[0]['content'] == "You are a helpful assistant"
            assert messages[1]['role'] == "user"
            assert messages[1]['content'] == "User prompt"
    
    def test_call_text_model_with_messages(self, mock_api_key, mock_openai_client):
        """Test call_text_model with messages list"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': mock_api_key}):
            with patch('llms.text.openai_text_model.OpenAI', return_value=mock_openai_client):
                model = OpenAITextModel()
            
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
            result = model.call_text_model(messages=messages)
            
            assert result == "Test response"
            call_args = mock_openai_client.chat.completions.create.call_args
            api_messages = call_args[1]['messages']
            assert len(api_messages) == 2
            assert api_messages == messages
    
    
    def test_call_text_model_empty_prompt(self, mock_api_key):
        """Test call_text_model with empty prompt"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': mock_api_key}):
            with patch('llms.text.openai_text_model.OpenAI'):
                model = OpenAITextModel()
            
            with pytest.raises(ValueError) as exc_info:
                model.call_text_model(prompt="")
            
            assert "Either 'prompt' or 'messages' must be provided" in str(exc_info.value)
    
    def test_call_text_model_no_prompt_no_messages(self, mock_api_key):
        """Test call_text_model without prompt or messages"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': mock_api_key}):
            with patch('llms.text.openai_text_model.OpenAI'):
                model = OpenAITextModel()
            
            with pytest.raises(ValueError) as exc_info:
                model.call_text_model()
            
            assert "Either 'prompt' or 'messages' must be provided" in str(exc_info.value)
    
    def test_call_text_model_invalid_messages_format(self, mock_api_key):
        """Test call_text_model with invalid messages format"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': mock_api_key}):
            with patch('llms.text.openai_text_model.OpenAI'):
                model = OpenAITextModel()
            
            with pytest.raises(Exception) as exc_info:
                model.call_text_model(messages=[{"invalid": "message"}])
            
            assert "Each message must be a dict with 'role' and 'content' keys" in str(exc_info.value)