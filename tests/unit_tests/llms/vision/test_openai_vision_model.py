"""
Tests for OpenAIVisionModel
"""
import pytest
import os
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src to path
# Calculate project root: go up from test file to project root
# test_openai_vision_model.py -> vision/ -> llms/ -> unit_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

from llms.vision.openai_vision_model import OpenAIVisionModel, OPENAI_AVAILABLE


@pytest.fixture
def mock_openai_client():
    """Creates a mock OpenAI client"""
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "Test vision response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_api_key():
    """Returns a mock API key"""
    return "test-api-key-12345"


@pytest.fixture
def sample_image_base64():
    """Returns a sample base64 encoded image"""
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI package not installed")
class TestOpenAIVisionModel:
    """Test class for OpenAIVisionModel"""
    
    def test_init_with_api_key(self, mock_api_key):
        """Test OpenAIVisionModel initialization with API key"""
        with patch.dict(os.environ, {}, clear=True):
            with patch('llms.vision.openai_vision_model.OpenAI') as mock_openai:
                model = OpenAIVisionModel(api_key=mock_api_key)
                
                assert model.api_key == mock_api_key
                assert model.model == "gpt-4o"
                assert model.default_max_tokens == 500
                assert model.default_temperature == 0.0
                mock_openai.assert_called_once_with(api_key=mock_api_key)
    
    def test_init_with_env_var(self, mock_api_key):
        """Test OpenAIVisionModel initialization with environment variable"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': mock_api_key}):
            with patch('llms.vision.openai_vision_model.OpenAI') as mock_openai:
                model = OpenAIVisionModel()
                
                assert model.api_key == mock_api_key
                mock_openai.assert_called_once_with(api_key=mock_api_key)
    
    def test_init_no_api_key(self):
        """Test OpenAIVisionModel initialization without API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                OpenAIVisionModel()
            
            assert "OpenAI API key is required" in str(exc_info.value)
    
    def test_init_custom_parameters(self, mock_api_key):
        """Test OpenAIVisionModel initialization with custom parameters"""
        with patch('llms.vision.openai_vision_model.OpenAI'):
            model = OpenAIVisionModel(
                api_key=mock_api_key,
                model="gpt-4-vision-preview",
                max_tokens=1000,
                temperature=0.5
            )
            
            assert model.model == "gpt-4-vision-preview"
            assert model.default_max_tokens == 1000
            assert model.default_temperature == 0.5
    
    def test_call_vision_model_with_single_image(self, mock_api_key, mock_openai_client, sample_image_base64):
        """Test call_vision_model with single image"""
        with patch('llms.vision.openai_vision_model.OpenAI', return_value=mock_openai_client):
            model = OpenAIVisionModel(api_key=mock_api_key)
            
            result = model.call_vision_model(
                prompt="Describe this image",
                images=sample_image_base64
            )
            
            assert result == "Test vision response"
            mock_openai_client.chat.completions.create.assert_called_once()
            
            # Verify call arguments
            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]['model'] == "gpt-4o"
            assert len(call_args[1]['messages']) == 1
            assert call_args[1]['messages'][0]['role'] == "user"
            content = call_args[1]['messages'][0]['content']
            assert len(content) == 2  # text + image
            assert content[0]['type'] == "text"
            assert content[0]['text'] == "Describe this image"
            assert content[1]['type'] == "image_url"
            assert "data:image/png;base64," in content[1]['image_url']['url']
    
    def test_call_vision_model_with_multiple_images(self, mock_api_key, mock_openai_client, sample_image_base64):
        """Test call_vision_model with multiple images"""
        with patch('llms.vision.openai_vision_model.OpenAI', return_value=mock_openai_client):
            model = OpenAIVisionModel(api_key=mock_api_key)
            
            images = [sample_image_base64, sample_image_base64]
            result = model.call_vision_model(
                prompt="Compare these images",
                images=images
            )
            
            assert result == "Test vision response"
            call_args = mock_openai_client.chat.completions.create.call_args
            content = call_args[1]['messages'][0]['content']
            # Should have 1 text + 2 images = 3 items
            assert len(content) == 3
            assert content[0]['type'] == "text"
            assert content[1]['type'] == "image_url"
            assert content[2]['type'] == "image_url"
    
    def test_call_vision_model_with_custom_max_tokens(self, mock_api_key, mock_openai_client, sample_image_base64):
        """Test call_vision_model with custom max_tokens"""
        with patch('llms.vision.openai_vision_model.OpenAI', return_value=mock_openai_client):
            model = OpenAIVisionModel(api_key=mock_api_key, max_tokens=1000)
            
            result = model.call_vision_model(
                prompt="Test",
                images=sample_image_base64,
                max_tokens=2000
            )
            
            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]['max_tokens'] == 2000
    
    def test_call_vision_model_with_custom_temperature(self, mock_api_key, mock_openai_client, sample_image_base64):
        """Test call_vision_model with custom temperature"""
        with patch('llms.vision.openai_vision_model.OpenAI', return_value=mock_openai_client):
            model = OpenAIVisionModel(api_key=mock_api_key, temperature=0.0)
            
            result = model.call_vision_model(
                prompt="Test",
                images=sample_image_base64,
                temperature=0.5
            )
            
            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]['temperature'] == 0.5
    
    def test_call_vision_model_with_additional_kwargs(self, mock_api_key, mock_openai_client, sample_image_base64):
        """Test call_vision_model with additional kwargs"""
        with patch('llms.vision.openai_vision_model.OpenAI', return_value=mock_openai_client):
            model = OpenAIVisionModel(api_key=mock_api_key)
            
            result = model.call_vision_model(
                prompt="Test",
                images=sample_image_base64,
                top_p=0.9,
                frequency_penalty=0.5
            )
            
            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]['top_p'] == 0.9
            assert call_args[1]['frequency_penalty'] == 0.5
    
    def test_call_vision_model_empty_prompt(self, mock_api_key, sample_image_base64):
        """Test call_vision_model with empty prompt"""
        with patch('llms.vision.openai_vision_model.OpenAI'):
            model = OpenAIVisionModel(api_key=mock_api_key)
            
            with pytest.raises(ValueError) as exc_info:
                model.call_vision_model(prompt="", images=sample_image_base64)
            
            assert "prompt cannot be empty" in str(exc_info.value)
    
    def test_call_vision_model_no_images(self, mock_api_key):
        """Test call_vision_model without images"""
        with patch('llms.vision.openai_vision_model.OpenAI'):
            model = OpenAIVisionModel(api_key=mock_api_key)
            
            with pytest.raises(ValueError) as exc_info:
                model.call_vision_model(prompt="Test", images=None)
            
            assert "images cannot be empty" in str(exc_info.value)
    
    def test_call_vision_model_empty_images_list(self, mock_api_key):
        """Test call_vision_model with empty images list"""
        with patch('llms.vision.openai_vision_model.OpenAI'):
            model = OpenAIVisionModel(api_key=mock_api_key)
            
            with pytest.raises(ValueError) as exc_info:
                model.call_vision_model(prompt="Test", images=[])
            
            assert "images cannot be empty" in str(exc_info.value)
    
    def test_prepare_image_data_with_data_url(self, mock_api_key):
        """Test _prepare_image_data with data URL format"""
        with patch('llms.vision.openai_vision_model.OpenAI'):
            model = OpenAIVisionModel(api_key=mock_api_key)
            
            data_url = "data:image/png;base64,iVBORw0KGgo="
            result = model._prepare_image_data(image_base64=data_url)
            
            assert result == data_url
    
    def test_prepare_image_data_with_raw_base64(self, mock_api_key):
        """Test _prepare_image_data with raw base64"""
        with patch('llms.vision.openai_vision_model.OpenAI'):
            model = OpenAIVisionModel(api_key=mock_api_key)
            
            raw_base64 = "iVBORw0KGgo="
            result = model._prepare_image_data(image_base64=raw_base64)
            
            assert result == "data:image/png;base64,iVBORw0KGgo="
    
    def test_call_vision_model_api_error(self, mock_api_key, sample_image_base64):
        """Test call_vision_model with API error"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        with patch('llms.vision.openai_vision_model.OpenAI', return_value=mock_client):
            model = OpenAIVisionModel(api_key=mock_api_key)
            
            with pytest.raises(Exception) as exc_info:
                model.call_vision_model(
                    prompt="Test",
                    images=sample_image_base64
                )
            
            assert "Error calling OpenAI Vision API" in str(exc_info.value)
    
    def test_call_vision_model_empty_response(self, mock_api_key, sample_image_base64):
        """Test call_vision_model with empty response"""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('llms.vision.openai_vision_model.OpenAI', return_value=mock_client):
            model = OpenAIVisionModel(api_key=mock_api_key)
            
            with pytest.raises(Exception) as exc_info:
                model.call_vision_model(
                    prompt="Test",
                    images=sample_image_base64
                )
            
            assert "OpenAI Vision API returned an empty response" in str(exc_info.value)
    
    def test_call_vision_model_response_stripped(self, mock_api_key, sample_image_base64):
        """Test that response is stripped"""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "  Response with spaces  "
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('llms.vision.openai_vision_model.OpenAI', return_value=mock_client):
            model = OpenAIVisionModel(api_key=mock_api_key)
            
            result = model.call_vision_model(
                prompt="Test",
                images=sample_image_base64
            )
            
            assert result == "Response with spaces"

