"""
Tests for BaseTextModel
"""
import pytest
from pathlib import Path
import sys
from abc import ABC

# Add src to path
# Calculate project root: go up from test file to project root
# test_base_text_model.py -> text/ -> llms/ -> unit_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

from llms.text.base_text_model import BaseTextModel


class MockTextModel(BaseTextModel):
    """Mock implementation of BaseTextModel for testing"""
    
    def __init__(self):
        self.call_count = 0
        self.last_prompt = None
        self.last_system_prompt = None
        self.last_messages = None
        self.last_kwargs = None
    
    def call_text_model(
        self,
        *,
        prompt: str,
        system_prompt: str = None,
        messages: list = None,
        **kwargs
    ) -> str:
        """Mock implementation of call_text_model"""
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        self.last_messages = messages
        self.last_kwargs = kwargs
        return f"Mock response for: {prompt}"


class TestBaseTextModel:
    """Test class for BaseTextModel"""
    
    def test_is_abstract_class(self):
        """Test that BaseTextModel is an abstract class"""
        assert issubclass(BaseTextModel, ABC)
        
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            BaseTextModel()
    
    def test_mock_implementation_works(self):
        """Test that a mock implementation can be instantiated"""
        model = MockTextModel()
        assert isinstance(model, BaseTextModel)
        assert model.call_count == 0
    
    def test_call_text_model_with_prompt(self):
        """Test call_text_model with a simple prompt"""
        model = MockTextModel()
        result = model.call_text_model(prompt="Test prompt")
        
        assert result == "Mock response for: Test prompt"
        assert model.call_count == 1
        assert model.last_prompt == "Test prompt"
        assert model.last_system_prompt is None
        assert model.last_messages is None
    
    def test_call_text_model_with_system_prompt(self):
        """Test call_text_model with system prompt"""
        model = MockTextModel()
        result = model.call_text_model(
            prompt="User prompt",
            system_prompt="System prompt"
        )
        
        assert result == "Mock response for: User prompt"
        assert model.last_system_prompt == "System prompt"
    
    def test_call_text_model_with_messages(self):
        """Test call_text_model with messages list"""
        model = MockTextModel()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        result = model.call_text_model(
            prompt="",
            messages=messages
        )
        
        assert model.last_messages == messages
    
    def test_call_text_model_with_kwargs(self):
        """Test call_text_model with additional kwargs"""
        model = MockTextModel()
        result = model.call_text_model(
            prompt="Test",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9
        )
        
        assert "max_tokens" in model.last_kwargs
        assert model.last_kwargs["max_tokens"] == 100
        assert model.last_kwargs["temperature"] == 0.7
        assert model.last_kwargs["top_p"] == 0.9
    
    def test_call_text_model_multiple_calls(self):
        """Test multiple calls to call_text_model"""
        model = MockTextModel()
        
        model.call_text_model(prompt="First")
        assert model.call_count == 1
        
        model.call_text_model(prompt="Second")
        assert model.call_count == 2
        
        model.call_text_model(prompt="Third")
        assert model.call_count == 3
        assert model.last_prompt == "Third"

