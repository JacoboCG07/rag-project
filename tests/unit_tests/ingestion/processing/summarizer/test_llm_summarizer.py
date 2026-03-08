"""
Tests for LLMSummarizer
$env:PYTHONPATH="$PWD"; pytest tests/unit_tests/ingestion/processing/summarizer
"""
import pytest
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add project root to path so that "src" is a package (same as production)
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingestion.processing.summarizer.llm_summarizer import LLMSummarizer
from src.llms.text.base_text_model import BaseTextModel


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
    with patch('src.ingestion.processing.summarizer.llm_summarizer.PromptLoader.read_file') as mock_read:
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
        assert summarizer_default.max_input_chars == 100_000
    
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
    
    def test_generate_summary_truncates_long_text(self, mock_text_model, mock_prompt_loader, caplog):
        """Test that text exceeding max_input_chars is truncated with warning"""
        summarizer = LLMSummarizer(
            text_model=mock_text_model,
            max_input_chars=50
        )
        long_text = "A" * 100 + ". Segunda oración."

        result = summarizer.generate_summary(long_text)

        assert result == "Mock summary response"
        assert mock_text_model.call_count == 1
        prompt_sent = mock_text_model.last_prompt
        assert "Texto recortado" in prompt_sent or "recortado" in prompt_sent.lower()
        assert len(prompt_sent) <= 50 + 100  # truncated content + suffix
        assert "Texto muy largo" in caplog.text or "Truncando" in caplog.text

    def test_generate_summary_short_text_not_truncated(self, mock_text_model, mock_prompt_loader):
        """Test that text under max_input_chars is passed unchanged"""
        summarizer = LLMSummarizer(
            text_model=mock_text_model,
            max_input_chars=1000
        )
        short_text = "Texto corto para resumir."

        result = summarizer.generate_summary(short_text)

        assert mock_text_model.last_prompt == short_text
        assert "Texto recortado" not in (mock_text_model.last_prompt or "")

    def test_get_summary_prompt(self, mock_prompt_loader):
        """Test that _get_summary_prompt loads the correct template"""
        text = "Test text"
        prompt, system_prompt = LLMSummarizer._get_summary_prompt(text)
        
        assert prompt == text
        assert system_prompt == "System prompt for summarization"
        assert isinstance(prompt, str)
        assert isinstance(system_prompt, str)
        mock_prompt_loader.assert_called_once()
        # El código usa Path(__file__).parent / "summarizer_prompt.md" → ruta absoluta
        call_path = mock_prompt_loader.call_args[0][0]
        assert call_path.replace("\\", "/").endswith("summarizer/summarizer_prompt.md")
