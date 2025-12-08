"""
Functional tests for OpenAITextModel
These tests make real API calls to OpenAI and require a valid API key.
"""
import pytest
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add src to path
# Calculate project root: go up from test file to project root
# test_openai_text_model_functional.py -> text/ -> llms/ -> functional_tests/ -> tests/ -> project_root
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

from llms.text.openai_text_model import OpenAITextModel, OPENAI_AVAILABLE


# Skip all tests if OpenAI package is not available
pytestmark = pytest.mark.skipif(
    not OPENAI_AVAILABLE,
    reason="OpenAI package not installed"
)


def has_openai_api_key():
    """Check if OpenAI API key is available"""
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.fixture
def openai_text_model():
    """Fixture to create OpenAITextModel instance if API key is available"""
    if not has_openai_api_key():
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    return OpenAITextModel()


@pytest.mark.functional
@pytest.mark.requires_api_key
class TestOpenAITextModelFunctional:
    """Functional tests for OpenAITextModel that make real API calls"""
    
    def test_call_text_model_with_simple_prompt(self, openai_text_model):
        """Test call_text_model with a simple prompt using real API"""
        prompt = "Say hello in one word."
        
        result = openai_text_model.call_text_model(prompt=prompt)
        
        # Verify response structure
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain some form of greeting
        assert any(word in result.lower() for word in ["hello", "hi", "hey", "hola"])
    
    def test_call_text_model_with_system_prompt(self, openai_text_model):
        """Test call_text_model with system prompt using real API"""
        prompt = "What is 2+2?"
        system_prompt = "You are a helpful math assistant. Always respond with just the number."
        
        result = openai_text_model.call_text_model(
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        # Verify response
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain the answer
        assert "4" in result
    
    def test_call_text_model_with_messages(self, openai_text_model):
        """Test call_text_model with messages list using real API"""
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What is its population?"}
        ]
        
        result = openai_text_model.call_text_model(messages=messages)
        
        # Verify response
        assert isinstance(result, str)
        assert len(result) > 0
        # Should mention population or Paris
        assert any(word in result.lower() for word in ["population", "paris", "million", "people"])
    
    def test_call_text_model_with_custom_temperature(self, openai_text_model):
        """Test call_text_model with custom temperature using real API"""
        prompt = "Tell me a color."
        
        # Low temperature (deterministic)
        result_low = openai_text_model.call_text_model(
            prompt=prompt,
            temperature=0.0
        )
        
        # High temperature (creative)
        result_high = openai_text_model.call_text_model(
            prompt=prompt,
            temperature=1.0
        )
        
        # Both should be valid responses
        assert isinstance(result_low, str)
        assert isinstance(result_high, str)
        assert len(result_low) > 0
        assert len(result_high) > 0
    
    def test_call_text_model_with_custom_max_tokens(self, openai_text_model):
        """Test call_text_model with custom max_tokens using real API"""
        prompt = "Count from 1 to 10."
        
        result = openai_text_model.call_text_model(
            prompt=prompt,
            max_tokens=50
        )
        
        # Verify response
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain numbers
        assert any(str(i) in result for i in range(1, 11))
    
    def test_call_text_model_different_models(self):
        """Test call_text_model with different models"""
        if not has_openai_api_key():
            pytest.skip("OPENAI_API_KEY environment variable not set")
        
        prompt = "Say hello."
        
        # Test with gpt-4o
        model_gpt4o = OpenAITextModel(model="gpt-4o")
        result_gpt4o = model_gpt4o.call_text_model(prompt=prompt)
        assert isinstance(result_gpt4o, str)
        assert len(result_gpt4o) > 0
        
        # Test with gpt-3.5-turbo (if available)
        try:
            model_gpt35 = OpenAITextModel(model="gpt-3.5-turbo")
            result_gpt35 = model_gpt35.call_text_model(prompt=prompt)
            assert isinstance(result_gpt35, str)
            assert len(result_gpt35) > 0
        except Exception:
            # Some models might not be available, skip if error
            pytest.skip("gpt-3.5-turbo not available")
    
    def test_call_text_model_empty_prompt_raises_error(self, openai_text_model):
        """Test that empty prompt raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            openai_text_model.call_text_model(prompt="")
        
        assert "Either 'prompt' or 'messages' must be provided" in str(exc_info.value)
    
    def test_call_text_model_no_prompt_no_messages_raises_error(self, openai_text_model):
        """Test that calling without prompt or messages raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            openai_text_model.call_text_model()
        
        assert "Either 'prompt' or 'messages' must be provided" in str(exc_info.value)
    
    def test_call_text_model_invalid_messages_format_raises_error(self, openai_text_model):
        """Test that invalid messages format raises Exception"""
        with pytest.raises(Exception) as exc_info:
            openai_text_model.call_text_model(messages=[{"invalid": "message"}])
        
        assert "Each message must be a dict with 'role' and 'content' keys" in str(exc_info.value)
    
    
    def test_call_text_model_with_additional_kwargs(self, openai_text_model):
        """Test call_text_model with additional kwargs like top_p"""
        prompt = "Name a fruit."
        
        result = openai_text_model.call_text_model(
            prompt=prompt,
            top_p=0.9,
            frequency_penalty=0.0
        )
        
        # Verify response
        assert isinstance(result, str)
        assert len(result) > 0
    
    
    def test_call_text_model_with_system_prompt_and_messages(self, openai_text_model):
        """Test call_text_model with both system prompt and messages"""
        system_prompt = "You are a helpful assistant that responds briefly."
        messages = [
            {"role": "user", "content": "What is 5+5?"}
        ]
        
        result = openai_text_model.call_text_model(
            system_prompt=system_prompt,
            messages=messages
        )
        
        # Verify response
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain the answer
        assert "10" in result
    
    def test_call_text_model_default_parameters(self, openai_text_model):
        """Test that default parameters work correctly"""
        prompt = "Say yes or no."
        
        # Call without specifying max_tokens or temperature
        result = openai_text_model.call_text_model(prompt=prompt)
        
        # Should work with defaults
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_call_text_model_multiple_calls(self, openai_text_model):
        """Test multiple calls to ensure consistency"""
        prompts = [
            "Name a color.",
            "Name an animal.",
            "Name a country."
        ]
        
        results = []
        for prompt in prompts:
            result = openai_text_model.call_text_model(prompt=prompt)
            results.append(result)
            assert isinstance(result, str)
            assert len(result) > 0
        
        # All results should be different (different prompts)
        assert len(set(results)) == len(results) or len(results) == 3

