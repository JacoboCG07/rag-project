"""
Tests for BaseVisionModel
"""
import pytest
from pathlib import Path
import sys
from abc import ABC

# Add src to path
# Calculate project root: go up from test file to project root
# test_base_vision_model.py -> vision/ -> llms/ -> unit_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

from llms.vision.base_vision_model import BaseVisionModel


class MockVisionModel(BaseVisionModel):
    """Mock implementation of BaseVisionModel for testing"""
    
    def __init__(self):
        self.call_count = 0
        self.last_prompt = None
        self.last_images = None
        self.last_kwargs = None
    
    def call_vision_model(
        self,
        *,
        prompt: str,
        images: list,
        **kwargs
    ) -> str:
        """Mock implementation of call_vision_model"""
        self.call_count += 1
        self.last_prompt = prompt
        self.last_images = images if isinstance(images, list) else [images]
        self.last_kwargs = kwargs
        return f"Mock response for: {prompt} with {len(self.last_images)} image(s)"


class TestBaseVisionModel:
    """Test class for BaseVisionModel"""
    
    def test_is_abstract_class(self):
        """Test that BaseVisionModel is an abstract class"""
        assert issubclass(BaseVisionModel, ABC)
        
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            BaseVisionModel()
    
    def test_mock_implementation_works(self):
        """Test that a mock implementation can be instantiated"""
        model = MockVisionModel()
        assert isinstance(model, BaseVisionModel)
        assert model.call_count == 0
    
    def test_call_vision_model_with_single_image(self):
        """Test call_vision_model with a single image"""
        model = MockVisionModel()
        image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        result = model.call_vision_model(
            prompt="Describe this image",
            images=image_base64
        )
        
        assert result == "Mock response for: Describe this image with 1 image(s)"
        assert model.call_count == 1
        assert model.last_prompt == "Describe this image"
        assert len(model.last_images) == 1
        assert model.last_images[0] == image_base64
    
    def test_call_vision_model_with_multiple_images(self):
        """Test call_vision_model with multiple images"""
        model = MockVisionModel()
        images = [
            "image1_base64",
            "image2_base64",
            "image3_base64"
        ]
        result = model.call_vision_model(
            prompt="Compare these images",
            images=images
        )
        
        assert result == "Mock response for: Compare these images with 3 image(s)"
        assert len(model.last_images) == 3
        assert model.last_images == images
    
    def test_call_vision_model_with_kwargs(self):
        """Test call_vision_model with additional kwargs"""
        model = MockVisionModel()
        image_base64 = "test_image_base64"
        result = model.call_vision_model(
            prompt="Test",
            images=image_base64,
            max_tokens=500,
            temperature=0.2,
            top_p=0.9
        )
        
        assert "max_tokens" in model.last_kwargs
        assert model.last_kwargs["max_tokens"] == 500
        assert model.last_kwargs["temperature"] == 0.2
        assert model.last_kwargs["top_p"] == 0.9
    
    def test_call_vision_model_multiple_calls(self):
        """Test multiple calls to call_vision_model"""
        model = MockVisionModel()
        image_base64 = "test_image"
        
        model.call_vision_model(prompt="First", images=image_base64)
        assert model.call_count == 1
        
        model.call_vision_model(prompt="Second", images=image_base64)
        assert model.call_count == 2
        
        model.call_vision_model(prompt="Third", images=image_base64)
        assert model.call_count == 3
        assert model.last_prompt == "Third"

