"""
Tests for LLMSummarizer
"""
import pytest
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src to path
# Calculate project root: go up from test file to project root
# test_llm_summarizer.py -> summarizer/ -> processing/ -> ingestion/ -> unit_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

from ingestion.processing.summarizer.llm_summarizer import LLMSummarizer
from llms.text.base_text_model import BaseTextModel


class MockTextModel(BaseTextModel):
    """Mock implementation of BaseTextModel for testing"""
    
    def __init__(self):
        self.call_count = 0
        self.last_prompt = None
        self.last_system_prompt = None
        self.last_max_tokens = None
        self.last_temperature = None
    
    def call_text_model(
        self,
        *,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """Mock implementation of call_text_model"""
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        self.last_max_tokens = max_tokens
        self.last_temperature = temperature
        return "Mock summary response"


@pytest.fixture
def mock_text_model():
    """Creates a mock text model"""
    return MockTextModel()


@pytest.fixture
def mock_prompt_loader():
    """Mocks PromptLoader.read_file"""
    with patch('ingestion.processing.summarizer.llm_summarizer.PromptLoader.read_file') as mock_read:
        mock_read.return_value = "System prompt for summarization"
        yield mock_read


class TestLLMSummarizer:
    """Test class for LLMSummarizer"""
    
    def test_init(self, mock_text_model, mock_prompt_loader):
        """Test LLMSummarizer initialization with valid and default parameters"""
        # Test with custom parameters
        summarizer = LLMSummarizer(
            text_model=mock_text_model,
            max_tokens=500,
            temperature=0.5
        )
        assert summarizer.text_model == mock_text_model
        assert summarizer.max_tokens == 500
        assert summarizer.temperature == 0.5
        
        # Test with default parameters
        summarizer_default = LLMSummarizer(text_model=mock_text_model)
        assert summarizer_default.max_tokens == 1_000
        assert summarizer_default.temperature == 0.3
    
    def test_init_with_invalid_text_model_raises_error(self, mock_prompt_loader):
        """Test that initialization with invalid text_model raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            LLMSummarizer(text_model=None)
        assert "text_model" in str(exc_info.value).lower()
        
        with pytest.raises(ValueError) as exc_info:
            LLMSummarizer(text_model="not a text model")
        assert "text_model" in str(exc_info.value).lower()
    
    
    def test_generate_summary_invalid_input_raises_error(self, mock_text_model, mock_prompt_loader):
        """Test that invalid input raises ValueError"""
        summarizer = LLMSummarizer(text_model=mock_text_model)
        
        # Test empty string
        with pytest.raises(ValueError) as exc_info:
            summarizer.generate_summary("")
        assert "non-empty string" in str(exc_info.value).lower()
        
        # Test None
        with pytest.raises(ValueError) as exc_info:
            summarizer.generate_summary(None)
        assert "non-empty string" in str(exc_info.value).lower()
        
        # Test whitespace only
        with pytest.raises(ValueError) as exc_info:
            summarizer.generate_summary("   \n\t  ")
        assert "empty after stripping" in str(exc_info.value).lower()
    
    def test_generate_summary_text_model_error_propagates(self, mock_text_model, mock_prompt_loader):
        """Test that errors from text_model are propagated"""
        mock_text_model.call_text_model = Mock(side_effect=Exception("API error"))
        summarizer = LLMSummarizer(text_model=mock_text_model)
        
        with pytest.raises(Exception) as exc_info:
            summarizer.generate_summary("Test text")
        
        assert "Error generating summary" in str(exc_info.value)
        assert "API error" in str(exc_info.value)
    
    def test_get_summary_prompt(self, mock_prompt_loader):
        """Test that _get_summary_prompt loads the correct template"""
        text = "Test text"
        prompt, system_prompt = LLMSummarizer._get_summary_prompt(text)
        
        assert prompt == text
        assert system_prompt == "System prompt for summarization"
        assert isinstance(prompt, str)
        assert isinstance(system_prompt, str)
        mock_prompt_loader.assert_called_once_with("src/ingestion/processing/summarizer/summarizer_prompt.md")
