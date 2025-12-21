"""
Functional tests for LLMImageDescriber
These tests make real API calls to vision model providers and require a valid API key.
"""
import pytest
import os
import base64
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add src to path
# Calculate project root: go up from test file to project root
# test_llm_image_describer_functional.py -> describer/ -> processing/ -> rag/ -> functional_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

# Load .env file from project root
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from rag.processing.describer.llm_image_describer import LLMImageDescriber
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
def llm_image_describer(openai_vision_model):
    """Fixture to create LLMImageDescriber instance with OpenAI vision model"""
    return LLMImageDescriber(vision_model=openai_vision_model)


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
class TestLLMImageDescriberFunctional:
    """Functional tests for LLMImageDescriber that make real API calls"""
    
    def test_describe_image_simple_image(self, llm_image_describer, sample_image_base64):
        """Test generating description for simple image using real API"""
        result = llm_image_describer.describe_image(image=sample_image_base64)
        
        # Verify response structure
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain some description
        assert len(result) > 10
    
    def test_describe_image_with_custom_max_tokens(self, openai_vision_model, sample_image_base64):
        """Test image description with custom max_tokens using real API"""
        describer = LLMImageDescriber(
            vision_model=openai_vision_model,
            max_tokens=50,
            temperature=0.3
        )
        
        result = describer.describe_image(image=sample_image_base64)
        
        # Verify response
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Verify that the description is relatively short (max_tokens=50)
        # Note: token count != character count, but we can check it's not extremely long
        word_count = len(result.split())
        assert word_count <= 100, (
            f"Description has {word_count} words, which seems too long for max_tokens=50. "
        )
    
    def test_describe_image_with_custom_prompt(self, llm_image_describer, sample_image_base64):
        """Test image description with custom prompt using real API"""
        custom_prompt = "Describe the colors in this image in one sentence."
        
        result = llm_image_describer.describe_image(
            image=sample_image_base64,
            prompt=custom_prompt
        )
        
        # Verify response
        assert isinstance(result, str)
        assert len(result) > 0
        # Should mention colors or visual elements
        assert any(word in result.lower() for word in ["color", "red", "pixel", "image", "see"])
    
    def test_describe_image_empty_image_raises_error(self, llm_image_describer):
        """Test that empty image raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            llm_image_describer.describe_image(image="")
        
        assert "non-empty" in str(exc_info.value).lower() or "cannot be empty" in str(exc_info.value).lower()
    
    def test_describe_image_none_image_raises_error(self, llm_image_describer):
        """Test that None image raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            llm_image_describer.describe_image(image=None)
        
        assert "must be provided" in str(exc_info.value).lower()
    
    def test_describe_image_multiple_images(self, llm_image_describer, sample_image_base64):
        """Test generating descriptions for multiple images"""
        images = [sample_image_base64, sample_image_base64]
        
        result = llm_image_describer.describe_image(image=images)
        
        # Verify response
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_describe_image_penguin_image(self, llm_image_describer, penguin_image_base64):
        """Test describing a real image (penguin) using real API"""
        result = llm_image_describer.describe_image(image=penguin_image_base64)
        
        # Verify response
        assert isinstance(result, str)
        assert len(result) > 0
        # Should mention penguin or bird-related terms
        result_lower = result.lower()
        assert any(word in result_lower for word in ["penguin", "bird", "animal", "black", "white", "image"])
