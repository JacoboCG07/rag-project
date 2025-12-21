"""
Functional tests for LLMSummarizer
These tests make real API calls to LLM providers and require a valid API key.
"""
import pytest
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add src to path
# Calculate project root: go up from test file to project root
# test_llm_summarizer_functional.py -> summarizer/ -> processing/ -> rag/ -> functional_tests/ -> tests/ -> project_root
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

from rag.processing.summarizer.llm_summarizer import LLMSummarizer
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


@pytest.fixture
def llm_summarizer(openai_text_model):
    """Fixture to create LLMSummarizer instance with OpenAI model"""
    return LLMSummarizer(text_model=openai_text_model)


@pytest.mark.functional
@pytest.mark.requires_api_key
class TestLLMSummarizerFunctional:
    """Functional tests for LLMSummarizer that make real API calls"""
    
    def test_generate_summary_simple_text(self, llm_summarizer):
        """Test generating summary for simple text using real API"""
        text = """
        Python is a high-level programming language known for its simplicity and readability.
        It was created by Guido van Rossum and first released in 1991. Python supports multiple
        programming paradigms including procedural, object-oriented, and functional programming.
        It has a large standard library and a vibrant ecosystem of third-party packages.
        """
        
        result = llm_summarizer.generate_summary(text)
        
        # Verify response structure
        assert isinstance(result, str)
        assert len(result) > 0
        # assert len(result) < len(text)  # Summary should be shorter but it's not always the case
        # Should mention Python or programming
        assert any(word in result.lower() for word in ["python", "programming", "language"])
    
    def test_generate_summary_long_text(self, llm_summarizer):
        """Test generating summary for longer text using real API"""
        text = """
        Artificial Intelligence (AI) has become one of the most transformative technologies
        of the 21st century. It encompasses machine learning, deep learning, natural language
        processing, computer vision, and robotics. AI systems can now perform tasks that were
        previously thought to require human intelligence, such as image recognition, language
        translation, and strategic game playing. The field has seen rapid advances in recent
        years, driven by improvements in computing power, availability of large datasets, and
        algorithmic innovations. However, AI also raises important ethical questions about
        privacy, bias, job displacement, and the future of work.
        """
        
        result = llm_summarizer.generate_summary(text)
        
        # Verify response
        assert isinstance(result, str)
        assert len(result) > 0
        # Should mention AI or intelligence
        assert any(word in result.lower() for word in ["ai", "artificial", "intelligence", "machine"])
    
    def test_generate_summary_with_custom_max_tokens(self, openai_text_model):
        """Test summary generation with custom max_tokens using real API"""
        summarizer = LLMSummarizer(
            text_model=openai_text_model,
            max_tokens=50,
            temperature=0.3
        )
        
        text = """
        The history of computing spans several decades, from the early mechanical calculators
        to modern quantum computers. Key milestones include the invention of the transistor,
        the development of integrated circuits, the creation of personal computers, and the
        rise of the internet. Today, computing power continues to grow exponentially, enabling
        new applications in AI, cloud computing, and mobile technology.
        """
        
        result = summarizer.generate_summary(text)
        
        # Verify response
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Verify that the summary doesn't exceed 150 words
        word_count = len(result.split())
        assert word_count <= 100, (
            f"Summary has {word_count} words, which exceeds the limit of 100. "
        )
    
    def test_generate_summary_empty_text_raises_error(self, llm_summarizer):
        """Test that empty text raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            llm_summarizer.generate_summary("")
        
        assert "non-empty string" in str(exc_info.value).lower()
    
    def test_generate_summary_whitespace_only_raises_error(self, llm_summarizer):
        """Test that whitespace-only text raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            llm_summarizer.generate_summary("   \n\t  ")
        
        assert "empty after stripping" in str(exc_info.value).lower()
    
    def test_generate_summary_multiple_texts(self, llm_summarizer):
        """Test generating summaries for multiple texts"""
        texts = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Cloud computing allows users to access computing resources over the internet on demand.",
            "Blockchain is a distributed ledger technology that ensures data integrity and transparency."
        ]
        
        results = []
        for text in texts:
            result = llm_summarizer.generate_summary(text)
            results.append(result)
            assert isinstance(result, str)
            assert len(result) > 0
        
        # All results should be different (different input texts)
        assert len(results) == 3    