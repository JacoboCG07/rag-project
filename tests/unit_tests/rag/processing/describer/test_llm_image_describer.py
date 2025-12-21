"""
Tests for LLMImageDescriber
"""
import pytest
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# Add src to path
# Calculate project root: go up from test file to project root
# test_llm_image_describer.py -> describer/ -> processing/ -> rag/ -> unit_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

from rag.processing.describer.llm_image_describer import LLMImageDescriber
from llms.vision.base_vision_model import BaseVisionModel


class MockVisionModel(BaseVisionModel):
    """Mock implementation of BaseVisionModel for testing"""
    
    def __init__(self):
        self.call_count = 0
        self.last_prompt = None
        self.last_images = None
        self.last_max_tokens = None
        self.last_temperature = None
    
    def call_vision_model(
        self,
        *,
        prompt: str,
        images,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """Mock implementation of call_vision_model"""
        self.call_count += 1
        self.last_prompt = prompt
        self.last_images = images
        self.last_max_tokens = max_tokens
        self.last_temperature = temperature
        return "Mock image description response"


@pytest.fixture
def mock_vision_model():
    """Creates a mock vision model"""
    return MockVisionModel()


@pytest.fixture
def mock_prompt_loader():
    """Mocks PromptLoader.read_file"""
    with patch('rag.processing.describer.llm_image_describer.PromptLoader.read_file') as mock_read:
        mock_read.return_value = "System prompt for image description"
        yield mock_read


class TestLLMImageDescriber:
    """Test class for LLMImageDescriber"""
    
    def test_init(self, mock_vision_model, mock_prompt_loader):
        """Test LLMImageDescriber initialization with valid and default parameters"""
        # Test with custom parameters
        describer = LLMImageDescriber(
            vision_model=mock_vision_model,
            max_tokens=300,
            temperature=0.5
        )
        assert describer.vision_model == mock_vision_model
        assert describer.max_tokens == 300
        assert describer.temperature == 0.5
        
        # Test with default parameters
        describer_default = LLMImageDescriber(vision_model=mock_vision_model)
        assert describer_default.max_tokens == 1000
        assert describer_default.temperature == 0.3
    
    def test_init_with_invalid_vision_model_raises_error(self, mock_prompt_loader):
        """Test that initialization with invalid vision_model raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            LLMImageDescriber(vision_model=None)
        assert "vision_model" in str(exc_info.value).lower()
        
        with pytest.raises(ValueError) as exc_info:
            LLMImageDescriber(vision_model="not a vision model")
        assert "vision_model" in str(exc_info.value).lower()
    
    def test_describe_image_success(self, mock_vision_model, mock_prompt_loader):
        """Test successful image description generation"""
        describer = LLMImageDescriber(
            vision_model=mock_vision_model,
            max_tokens=300,
            temperature=0.3
        )
        
        image_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        result = describer.describe_image(image=image_base64)
        
        assert result == "Mock image description response"
        assert mock_vision_model.call_count == 1
        assert mock_vision_model.last_prompt == "System prompt for image description"
        assert mock_vision_model.last_images == image_base64
        assert mock_vision_model.last_max_tokens == 300
        assert mock_vision_model.last_temperature == 0.3
        mock_prompt_loader.assert_called_once_with("src/rag/processing/describer/image_describer_prompt.md")
    
    def test_describe_image_with_custom_prompt(self, mock_vision_model, mock_prompt_loader):
        """Test image description with custom prompt"""
        describer = LLMImageDescriber(vision_model=mock_vision_model)
        
        image_base64 = "data:image/png;base64,test123"
        custom_prompt = "What colors do you see?"
        result = describer.describe_image(image=image_base64, prompt=custom_prompt)
        
        assert mock_vision_model.last_prompt == custom_prompt
        assert result == "Mock image description response"
    
    def test_describe_image_with_multiple_images(self, mock_vision_model, mock_prompt_loader):
        """Test image description with multiple images"""
        describer = LLMImageDescriber(vision_model=mock_vision_model)
        
        images = [
            "data:image/png;base64,image1",
            "data:image/png;base64,image2"
        ]
        result = describer.describe_image(image=images)
        
        assert mock_vision_model.last_images == images
        assert result == "Mock image description response"
    
    def test_describe_image_invalid_input_raises_error(self, mock_vision_model, mock_prompt_loader):
        """Test that invalid input raises ValueError"""
        describer = LLMImageDescriber(vision_model=mock_vision_model)
        
        # Test None
        with pytest.raises(ValueError) as exc_info:
            describer.describe_image(image=None)
        assert "must be provided" in str(exc_info.value).lower()
        
        # Test empty string
        with pytest.raises(ValueError) as exc_info:
            describer.describe_image(image="")
        assert "non-empty" in str(exc_info.value).lower() or "cannot be empty" in str(exc_info.value).lower()
        
        # Test empty list
        with pytest.raises(ValueError) as exc_info:
            describer.describe_image(image=[])
        assert "non-empty" in str(exc_info.value).lower() or "cannot be empty" in str(exc_info.value).lower()
        
        # Test list with empty strings
        with pytest.raises(ValueError) as exc_info:
            describer.describe_image(image=["", ""])
        assert "non-empty" in str(exc_info.value).lower() or "cannot be empty" in str(exc_info.value).lower()
    
    def test_describe_image_vision_model_error_propagates(self, mock_vision_model, mock_prompt_loader):
        """Test that errors from vision_model are propagated"""
        mock_vision_model.call_vision_model = Mock(side_effect=Exception("API error"))
        describer = LLMImageDescriber(vision_model=mock_vision_model)
        
        with pytest.raises(Exception) as exc_info:
            describer.describe_image(image="data:image/png;base64,test")
        
        assert "Error generating image description" in str(exc_info.value)
        assert "API error" in str(exc_info.value)
    
    def test_get_description_prompt(self, mock_prompt_loader):
        """Test that _get_description_prompt loads the correct template"""
        prompt = LLMImageDescriber._get_description_prompt()
        
        assert prompt == "System prompt for image description"
        assert isinstance(prompt, str)
        mock_prompt_loader.assert_called_once_with("src/rag/processing/describer/image_describer_prompt.md")
