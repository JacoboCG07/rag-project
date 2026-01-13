"""
Functional tests for LLMDocumentChooser
These tests make real API calls to LLM providers and require a valid API key.
"""
import pytest
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add project root and src to path
# Calculate project root: go up from test file to project root
# test_llm_chooser_functional.py -> chooser/ -> document_selection/ -> search/ -> functional_tests/ -> tests/ -> project_root
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent.parent.parent
src_path = project_root / "src"
if project_root.exists():
    sys.path.insert(0, str(project_root))
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

# Load .env file from project root
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Import using importlib to avoid __init__.py issues
import importlib.util
llm_chooser_path = src_path / "search" / "document_selection" / "chooser" / "llm_chooser.py"
spec = importlib.util.spec_from_file_location("llm_chooser", llm_chooser_path)
llm_chooser_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_chooser_module)
LLMDocumentChooser = llm_chooser_module.LLMDocumentChooser

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
    
    return OpenAITextModel(model="gpt-4o-mini")


@pytest.fixture
def llm_chooser(openai_text_model):
    """Fixture to create LLMDocumentChooser instance with OpenAI model"""
    return LLMDocumentChooser(
        text_model=openai_text_model,
        max_tokens=500,
        temperature=0.2
    )


@pytest.fixture
def sample_summaries():
    """Fixture with sample document summaries for testing"""
    return [
        {
            "file_id": "doc_001",
            "file_name": "manual_instalacion.pdf",
            "type_file": "PDF",
            "total_pages": "50",
            "total_chapters": "5",
            "total_num_image": "10",
            "text": "Manual completo de instalación del sistema. Incluye requisitos, pasos detallados y solución de problemas comunes."
        },
        {
            "file_id": "doc_002",
            "file_name": "guia_usuario.pdf",
            "type_file": "PDF",
            "total_pages": "120",
            "total_chapters": "12",
            "total_num_image": "45",
            "text": "Guía de usuario completa con todas las funcionalidades del sistema. Incluye ejemplos prácticos y casos de uso."
        },
        {
            "file_id": "doc_003",
            "file_name": "api_reference.pdf",
            "type_file": "PDF",
            "total_pages": "200",
            "total_chapters": "15",
            "total_num_image": "30",
            "text": "Referencia completa de la API del sistema. Documentación técnica con ejemplos de código y endpoints."
        },
        {
            "file_id": "doc_004",
            "file_name": "troubleshooting.pdf",
            "type_file": "PDF",
            "total_pages": "80",
            "total_chapters": "8",
            "total_num_image": "20",
            "text": "Guía de solución de problemas. Errores comunes, diagnósticos y soluciones paso a paso."
        },
        {
            "file_id": "doc_005",
            "file_name": "arquitectura.pdf",
            "type_file": "PDF",
            "total_pages": "150",
            "total_chapters": "10",
            "total_num_image": "60",
            "text": "Documentación de arquitectura del sistema. Diseño, componentes, flujos de datos y decisiones técnicas."
        }
    ]


@pytest.fixture
def sample_markdown(sample_summaries):
    """Fixture with sample markdown descriptions"""
    formatter_path = src_path / "search" / "document_selection" / "formatter.py"
    spec = importlib.util.spec_from_file_location("formatter", formatter_path)
    formatter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(formatter_module)
    MarkdownGenerator = formatter_module.MarkdownGenerator
    
    generator = MarkdownGenerator()
    return generator.generate_all_documents_markdown(sample_summaries)


@pytest.mark.functional
@pytest.mark.requires_api_key
class TestLLMDocumentChooserFunctional:
    """Functional tests for LLMDocumentChooser that make real API calls"""
    
    def test_initialization_with_valid_model(self, openai_text_model):
        """Test that LLMDocumentChooser initializes correctly with valid model"""
        chooser = LLMDocumentChooser(
            text_model=openai_text_model,
            max_tokens=500,
            temperature=0.2
        )
        
        assert chooser.text_model == openai_text_model
        assert chooser.max_tokens == 500
        assert chooser.temperature == 0.2
    
    def test_initialization_without_model_raises_error(self):
        """Test that initialization without model raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            LLMDocumentChooser(text_model=None)
        
        assert "text_model" in str(exc_info.value).lower()
        assert "BaseTextModel" in str(exc_info.value)
    
    def test_initialization_with_invalid_model_raises_error(self):
        """Test that initialization with invalid model type raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            LLMDocumentChooser(text_model="not_a_model")
        
        assert "BaseTextModel" in str(exc_info.value)
    
    def test_choose_documents_basic(self, llm_chooser, sample_markdown, sample_summaries):
        """Test basic document selection with real API call"""
        user_query = "Necesito documentación sobre instalación del sistema"
        
        selected_ids = llm_chooser.choose_documents(
            markdown_descriptions=sample_markdown,
            user_query=user_query,
            summaries=sample_summaries
        )
        
        # Verify response structure
        assert isinstance(selected_ids, list)
        assert len(selected_ids) > 0
        assert len(selected_ids) <= len(sample_summaries)
        
        # Verify all selected IDs are valid
        valid_ids = {s["file_id"] for s in sample_summaries}
        for selected_id in selected_ids:
            assert selected_id in valid_ids, f"Selected ID {selected_id} is not in valid IDs"
        
        # Should select doc_001 (manual_instalacion) for installation query
        assert "doc_001" in selected_ids, "Should select installation manual for installation query"
    
    def test_choose_documents_api_reference_query(self, llm_chooser, sample_markdown, sample_summaries):
        """Test document selection for API reference query"""
        user_query = "Busco documentación técnica de la API"
        
        selected_ids = llm_chooser.choose_documents(
            markdown_descriptions=sample_markdown,
            user_query=user_query,
            summaries=sample_summaries
        )
        
        assert isinstance(selected_ids, list)
        assert len(selected_ids) > 0
        
        # Should select doc_003 (api_reference) for API query
        assert "doc_003" in selected_ids, "Should select API reference for API query"
    
    def test_choose_documents_with_details(self, llm_chooser, sample_markdown, sample_summaries):
        """Test choose_documents_with_details returns complete information"""
        user_query = "Guía de usuario y manuales"
        
        selected_docs = llm_chooser.choose_documents_with_details(
            markdown_descriptions=sample_markdown,
            user_query=user_query,
            summaries=sample_summaries
        )
        
        # Verify response structure
        assert isinstance(selected_docs, list)
        assert len(selected_docs) > 0
        
        # Verify each document has all required fields
        for doc in selected_docs:
            assert "file_id" in doc
            assert "file_name" in doc
            assert "type_file" in doc
            assert "text" in doc
            assert doc["file_id"] in [s["file_id"] for s in sample_summaries]
    
    def test_choose_documents_empty_markdown_raises_error(self, llm_chooser, sample_summaries):
        """Test that empty markdown raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            llm_chooser.choose_documents(
                markdown_descriptions="",
                user_query="test query",
                summaries=sample_summaries
            )
        
        assert "markdown_descriptions" in str(exc_info.value).lower()
    
    def test_choose_documents_empty_query_raises_error(self, llm_chooser, sample_markdown, sample_summaries):
        """Test that empty query raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            llm_chooser.choose_documents(
                markdown_descriptions=sample_markdown,
                user_query="",
                summaries=sample_summaries
            )
        
        assert "user_query" in str(exc_info.value).lower()
    
    def test_choose_documents_whitespace_only_query_raises_error(self, llm_chooser, sample_markdown, sample_summaries):
        """Test that whitespace-only query raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            llm_chooser.choose_documents(
                markdown_descriptions=sample_markdown,
                user_query="   \n\t  ",
                summaries=sample_summaries
            )
        
        assert "empty after strip" in str(exc_info.value).lower()
    
    def test_choose_documents_invalid_markdown_type_raises_error(self, llm_chooser, sample_summaries):
        """Test that non-string markdown raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            llm_chooser.choose_documents(
                markdown_descriptions=None,
                user_query="test query",
                summaries=sample_summaries
            )
        
        assert "markdown_descriptions" in str(exc_info.value).lower()
    
    def test_choose_documents_invalid_query_type_raises_error(self, llm_chooser, sample_markdown, sample_summaries):
        """Test that non-string query raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            llm_chooser.choose_documents(
                markdown_descriptions=sample_markdown,
                user_query=None,
                summaries=sample_summaries
            )
        
        assert "user_query" in str(exc_info.value).lower()
    
    def test_choose_documents_custom_temperature(self, openai_text_model, sample_markdown, sample_summaries):
        """Test document selection with custom temperature"""
        chooser = LLMDocumentChooser(
            text_model=openai_text_model,
            max_tokens=500,
            temperature=0.5  # Higher temperature
        )
        
        user_query = "Documentación técnica"
        
        selected_ids = chooser.choose_documents(
            markdown_descriptions=sample_markdown,
            user_query=user_query,
            summaries=sample_summaries
        )
        
        assert isinstance(selected_ids, list)
        assert len(selected_ids) > 0
    
    def test_choose_documents_custom_max_tokens(self, openai_text_model, sample_markdown, sample_summaries):
        """Test document selection with custom max_tokens"""
        chooser = LLMDocumentChooser(
            text_model=openai_text_model,
            max_tokens=200,  # Lower max tokens
            temperature=0.2
        )
        
        user_query = "Manuales de usuario"
        
        selected_ids = chooser.choose_documents(
            markdown_descriptions=sample_markdown,
            user_query=user_query,
            summaries=sample_summaries
        )
        
        assert isinstance(selected_ids, list)
        assert len(selected_ids) > 0
    
    def test_parse_response_with_valid_ids(self, llm_chooser, sample_summaries):
        """Test parsing LLM response with valid file IDs"""
        # Simulate LLM response with valid IDs
        response = "doc_001, doc_002, doc_003"
        
        selected_ids = llm_chooser._parse_response(response, sample_summaries)
        
        assert isinstance(selected_ids, list)
        assert len(selected_ids) == 3
        assert "doc_001" in selected_ids
        assert "doc_002" in selected_ids
        assert "doc_003" in selected_ids
    
    def test_parse_response_with_invalid_ids(self, llm_chooser, sample_summaries):
        """Test parsing LLM response with invalid file IDs"""
        # Simulate LLM response with invalid IDs
        response = "invalid_id_1, invalid_id_2, doc_001"
        
        selected_ids = llm_chooser._parse_response(response, sample_summaries)
        
        # Should only return valid IDs
        assert isinstance(selected_ids, list)
        assert "doc_001" in selected_ids
        assert "invalid_id_1" not in selected_ids
        assert "invalid_id_2" not in selected_ids
    
    def test_parse_response_with_newlines(self, llm_chooser, sample_summaries):
        """Test parsing LLM response with newline-separated IDs"""
        response = "doc_001\ndoc_002\ndoc_003"
        
        selected_ids = llm_chooser._parse_response(response, sample_summaries)
        
        assert isinstance(selected_ids, list)
        assert len(selected_ids) == 3
    
    def test_parse_response_empty_response(self, llm_chooser, sample_summaries):
        """Test parsing empty LLM response"""
        response = ""
        
        selected_ids = llm_chooser._parse_response(response, sample_summaries)
        
        assert isinstance(selected_ids, list)
        assert len(selected_ids) == 0
    
    def test_get_chooser_prompt_structure(self, sample_markdown):
        """Test that prompt structure is correct"""
        user_query = "Test query"
        
        prompt, system_prompt = LLMDocumentChooser._get_chooser_prompt(
            sample_markdown,
            user_query
        )
        
        assert isinstance(prompt, str)
        assert isinstance(system_prompt, str)
        assert user_query in prompt
        assert "Documentos disponibles" in prompt
        assert len(system_prompt) > 0

