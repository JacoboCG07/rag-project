"""
Functional tests for OpenAIVisionModel
These tests make real API calls to OpenAI and require a valid API key.
"""
import pytest
import os
import base64
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add src to path
# Calculate project root: go up from test file to project root
# test_openai_vision_model_functional.py -> vision/ -> llms/ -> functional_tests/ -> tests/ -> project_root
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

from llms.vision.openai_vision_model import OpenAIVisionModel, OPENAI_AVAILABLE


# Skip all tests if OpenAI package is not available
pytestmark = pytest.mark.skipif(
    not OPENAI_AVAILABLE,
    reason="OpenAI package not installed"
)


def has_openai_api_key():
    """Check if OpenAI API key is available"""
    return bool(os.getenv("OPENAI_API_KEY"))


def create_simple_test_image_base64():
    """Create a simple 1x1 pixel PNG image in base64 format for testing"""
    # Minimal valid PNG image (1x1 red pixel)
    # PNG signature + IHDR + IDAT + IEND chunks
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    return base64.b64encode(png_data).decode('utf-8')


def load_image_as_base64(image_path: Path) -> str:
    """Load an image file and convert it to base64 string"""
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        return base64.b64encode(image_data).decode('utf-8')


@pytest.fixture
def openai_vision_model():
    """Fixture to create OpenAIVisionModel instance if API key is available"""
    if not has_openai_api_key():
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    return OpenAIVisionModel()


@pytest.fixture
def sample_image_base64():
    """Fixture to provide a sample base64 encoded image (simple test image)"""
    return create_simple_test_image_base64()


@pytest.fixture
def penguin_image_base64():
    """Fixture to provide the penguin image from fixtures as base64"""
    fixtures_path = project_root / "tests" / "fixtures" / "penguin.webp"
    
    if not fixtures_path.exists():
        pytest.skip(f"Penguin image not found at {fixtures_path}")
    
    return load_image_as_base64(fixtures_path)


@pytest.mark.functional
@pytest.mark.requires_api_key
class TestOpenAIVisionModelFunctional:
    """Functional tests for OpenAIVisionModel that make real API calls"""
    
    def test_call_vision_model_with_single_image(self, openai_vision_model, sample_image_base64):
        """Test call_vision_model with a single image using real API"""
        prompt = "Describe what you see in this image in one sentence."
        
        result = openai_vision_model.call_vision_model(
            prompt=prompt,
            images=sample_image_base64
        )
        
        # Verify response structure
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be a description of some kind
        assert len(result.split()) > 0
    
    def test_call_vision_model_with_multiple_images(self, openai_vision_model, sample_image_base64):
        """Test call_vision_model with multiple images using real API"""
        prompt = "What do you see in these images?"
        images = [sample_image_base64, sample_image_base64]
        
        result = openai_vision_model.call_vision_model(
            prompt=prompt,
            images=images
        )
        
        # Verify response
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_call_vision_model_with_custom_parameters(self, openai_vision_model, sample_image_base64):
        """Test call_vision_model with custom max_tokens and temperature"""
        prompt = "What color is in this image?"
        
        result = openai_vision_model.call_vision_model(
            prompt=prompt,
            images=sample_image_base64,
            max_tokens=100,
            temperature=0.5
        )
        
        # Verify response
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_call_vision_model_empty_prompt_raises_error(self, openai_vision_model, sample_image_base64):
        """Test that empty prompt raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            openai_vision_model.call_vision_model(
                prompt="",
                images=sample_image_base64
            )
        
        assert "prompt cannot be empty" in str(exc_info.value)
    
    def test_call_vision_model_no_images_raises_error(self, openai_vision_model):
        """Test that calling without images raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            openai_vision_model.call_vision_model(
                prompt="Describe this image",
                images=None
            )
        
        assert "images cannot be empty" in str(exc_info.value)
    
    def test_call_vision_model_with_data_url_format(self, openai_vision_model, sample_image_base64):
        """Test call_vision_model with data URL format image"""
        # Convert base64 to data URL format
        data_url = f"data:image/png;base64,{sample_image_base64}"
        prompt = "What do you see?"
        
        result = openai_vision_model.call_vision_model(
            prompt=prompt,
            images=data_url
        )
        
        # Verify response
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_call_vision_model_describe_penguin(self, openai_vision_model, penguin_image_base64):
        """Test call_vision_model with real penguin image and verify it describes a penguin"""
        prompt = "Describe what you see in this image in detail."
        
        result = openai_vision_model.call_vision_model(
            prompt=prompt,
            images=penguin_image_base64
        )
        
        # Verify response structure
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Verify that the response mentions penguin or related terms
        result_lower = result.lower()
        assert any(word in result_lower for word in [
            "penguin", "penguins", "bird", "antarctic", "antarctica", 
            "black", "white", "animal", "wildlife"
        ]), f"Response should mention penguin-related terms. Got: {result}"

