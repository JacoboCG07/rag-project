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
# test_llm_chooser_functional.py -> chooser/ -> document_selection/ -> retrieval/ -> functional_tests/ -> tests/ -> project_root
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
llm_chooser_path = src_path / "retrieval" / "document_selection" / "chooser" / "llm_chooser.py"
spec = importlib.util.spec_from_file_location("llm_chooser", llm_chooser_path)
llm_chooser_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_chooser_module)
LLMDocumentChooser = llm_chooser_module.LLMDocumentChooser

from src.llms.text.openai_text_model import OpenAITextModel, OPENAI_AVAILABLE


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
            "file_type": "PDF",
            "pages": "50",
            "chapters": "5",
            "full_images": "10",
            "text": "Manual completo de instalación del sistema. Incluye requisitos, pasos detallados y solución de problemas comunes."
        },
        {
            "file_id": "doc_002",
            "file_name": "guia_usuario.pdf",
            "file_type": "PDF",
            "pages": "120",
            "chapters": "12",
            "full_images": "45",
            "text": "Guía de usuario completa con todas las funcionalidades del sistema. Incluye ejemplos prácticos y casos de uso."
        },
        {
            "file_id": "doc_003",
            "file_name": "api_reference.pdf",
            "file_type": "PDF",
            "pages": "200",
            "chapters": "15",
            "full_images": "30",
            "text": "Referencia completa de la API del sistema. Documentación técnica con ejemplos de código y endpoints."
        },
        {
            "file_id": "doc_004",
            "file_name": "troubleshooting.pdf",
            "file_type": "PDF",
            "pages": "80",
            "chapters": "8",
            "full_images": "20",
            "text": "Guía de solución de problemas. Errores comunes, diagnósticos y soluciones paso a paso."
        },
        {
            "file_id": "doc_005",
            "file_name": "arquitectura.pdf",
            "file_type": "PDF",
            "pages": "150",
            "chapters": "10",
            "full_images": "60",
            "text": "Documentación de arquitectura del sistema. Diseño, componentes, flujos de datos y decisiones técnicas."
        }
    ]


@pytest.fixture
def sample_markdown(sample_summaries):
    """Fixture with sample markdown descriptions"""
    formatter_path = src_path / "retrieval" / "common" / "formatter.py"
    spec = importlib.util.spec_from_file_location("formatter", formatter_path)
    formatter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(formatter_module)
    MarkdownGenerator = formatter_module.MarkdownGenerator

    generator = MarkdownGenerator()
    return generator.generate_all_documents_markdown(sample_summaries)


# IDs y datos del escenario CV recruitment (propuesta + 4 CVs)
JOB_PROPOSAL_ID = "9b88a532626e4e2efdc76cbe0e97b1c1"
CV_CANDIDATE_1_ID = "dd05cabc39f2920568d6e34fec7c3639"
CV_CANDIDATE_2_ID = "4a8bc89a3d3308679cc0fe7a93042048"
CV_CANDIDATE_3_ID = "7f0d024740a44d675d9604f3dbf26b62"
CV_CANDIDATE_4_ID = "3d3d62350b58ad272721a90822a20edc"


@pytest.fixture
def cv_recruitment_summaries():
    """Summaries para el escenario de reclutamiento: 1 propuesta + 4 CVs."""
    return [
        {
            "file_id": JOB_PROPOSAL_ID,
            "file_name": "summary_job_proposal.pdf",
            "file_type": "PDF",
            "pages": "1",
            "chapters": "false",
            "full_images": "0",
            "text": "Propuesta de trabajo Desarrollador Backend Python, Madrid, requisitos técnicos."
        },
        {
            "file_id": CV_CANDIDATE_1_ID,
            "file_name": "summary_cv_candidate_1.pdf",
            "file_type": "PDF",
            "pages": "1",
            "chapters": "false",
            "full_images": "0",
            "text": "CV Alberto Ruiz Martínez, desarrollador backend Python."
        },
        {
            "file_id": CV_CANDIDATE_2_ID,
            "file_name": "summary_cv_candidate_2.pdf",
            "file_type": "PDF",
            "pages": "1",
            "chapters": "false",
            "full_images": "0",
            "text": "CV Carlos Méndez Ortega, área comercial y ventas."
        },
        {
            "file_id": CV_CANDIDATE_3_ID,
            "file_name": "summary_cv_candidate_3.pdf",
            "file_type": "PDF",
            "pages": "1",
            "chapters": "false",
            "full_images": "0",
            "text": "CV Laura Fernández Gómez, desarrolladora frontend."
        },
        {
            "file_id": CV_CANDIDATE_4_ID,
            "file_name": "summary_cv_candidate_4.pdf",
            "file_type": "PDF",
            "pages": "1",
            "chapters": "false",
            "full_images": "0",
            "text": "CV Manuel García López, desarrollador backend senior."
        },
    ]


@pytest.fixture
def cv_recruitment_markdown():
    """Markdown de documentos para el escenario CV recruitment (propuesta + 4 CVs)."""
    return """# 📚 Biblioteca de Documentos

Total de documentos: **5**

---
## 📄 summary_job_proposal.pdf

- **ID:** `9b88a532626e4e2efdc76cbe0e97b1c1`
- **Tipo:** PDF
- **Capítulos:** false
- **Páginas:** 1
- **Imágenes:** 0

### Descripción:
**Tema Principal**: La oferta de empleo para un Desarrollador Backend especializado en Python.

**Descripción del Contenido**: El documento es una propuesta de trabajo que detalla una vacante para un Desarrollador Backend con experiencia en Python, ubicada en Madrid con modalidad híbrida. Se especifican las responsabilidades del puesto, que incluyen el desarrollo y mantenimiento de servicios backend, diseño de APIs REST, y trabajo con bases de datos relacionales y NoSQL. También se mencionan los requisitos técnicos obligatorios y valorables, como el manejo de Django, FastAPI, y experiencia en entornos cloud.

---
## 📄 summary_cv_candidate_1.pdf

- **ID:** `dd05cabc39f2920568d6e34fec7c3639`
- **Tipo:** PDF
- **Capítulos:** false
- **Páginas:** 1
- **Imágenes:** 0

### Descripción:
**Tema Principal**: CV de Alberto Ruiz Martínez, desarrollador backend junior/mid con experiencia en Python.

---
## 📄 summary_cv_candidate_2.pdf

- **ID:** `4a8bc89a3d3308679cc0fe7a93042048`
- **Tipo:** PDF
- **Capítulos:** false
- **Páginas:** 1
- **Imágenes:** 0

### Descripción:
**Tema Principal**: CV de Carlos Méndez Ortega, profesional del área comercial con experiencia en ventas.

---
## 📄 summary_cv_candidate_3.pdf

- **ID:** `7f0d024740a44d675d9604f3dbf26b62`
- **Tipo:** PDF
- **Capítulos:** false
- **Páginas:** 1
- **Imágenes:** 0

### Descripción:
**Tema Principal**: CV de Laura Fernández Gómez, desarrolladora frontend con experiencia en interfaces web.

---
## 📄 summary_cv_candidate_4.pdf

- **ID:** `3d3d62350b58ad272721a90822a20edc`
- **Tipo:** PDF
- **Capítulos:** false
- **Páginas:** 1
- **Imágenes:** 0

### Descripción:
**Tema Principal**: CV de Manuel García López, desarrollador backend senior especializado en Python.

---
"""


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
    
    def test_choose_documents_enriches_to_full_summaries(self, llm_chooser, sample_markdown, sample_summaries):
        """Tras elegir file_ids, los summaries completos conservan file_id, nombre, tipo y texto."""
        user_query = "Guía de usuario y manuales"

        selected_ids = llm_chooser.choose_documents(
            markdown_descriptions=sample_markdown,
            user_query=user_query,
            summaries=sample_summaries,
        )
        id_set = set(selected_ids)
        selected_docs = [s for s in sample_summaries if s.get("file_id") in id_set]

        assert isinstance(selected_docs, list)
        assert len(selected_docs) > 0

        for doc in selected_docs:
            assert "file_id" in doc
            assert "file_name" in doc
            assert "file_type" in doc
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

    def test_choose_documents_cv_recruitment_evaluate_all_candidates(
        self, llm_chooser, cv_recruitment_markdown, cv_recruitment_summaries
    ):
        """Test selección con consulta de evaluar todos los candidatos según la propuesta.
        El chooser debe incluir la propuesta de trabajo y los 4 CVs."""
        user_query = (
            "Evalua absolutamente todos los candidatos teniendo en cuenta los requisitos técnicos "
            "de la propuesta de trabajo y ponles una nota de 0 a 10, y luego ordénalos de mayor a menor nota."
        )
        selected_ids = llm_chooser.choose_documents(
            markdown_descriptions=cv_recruitment_markdown,
            user_query=user_query,
            summaries=cv_recruitment_summaries,
        )
        valid_ids = {s["file_id"] for s in cv_recruitment_summaries}
        assert isinstance(selected_ids, list), "Debe devolver una lista de IDs"
        assert len(selected_ids) > 0, "Debe seleccionar al menos un documento"
        for sid in selected_ids:
            assert sid in valid_ids, f"ID seleccionado {sid} no está en los documentos disponibles"
        # Debe incluir la propuesta (requisitos del puesto)
        assert JOB_PROPOSAL_ID in selected_ids, (
            "La consulta pide evaluar según la propuesta: debe incluir el documento de la propuesta de trabajo"
        )
        # Debe incluir al menos 2 CVs (el modelo puede excluir CVs claramente no relevantes, pero debe incluir varios)
        cv_ids = {CV_CANDIDATE_1_ID, CV_CANDIDATE_2_ID, CV_CANDIDATE_3_ID, CV_CANDIDATE_4_ID}
        selected_cvs = cv_ids & set(selected_ids)
        assert len(selected_cvs) >= 2, (
            f"La consulta pide evaluar candidatos: debe incluir al menos 2 CVs (seleccionados: {selected_cvs})"
        )

    def test_parse_response_with_valid_ids(self, llm_chooser, sample_summaries):
        """Test parsing LLM response with valid file IDs"""
        # Simulate LLM response with valid IDs
        response = "doc_001, doc_002, doc_003"
        
        selected_ids, think = llm_chooser._parse_response(response, sample_summaries)
        
        assert isinstance(selected_ids, list)
        assert len(selected_ids) == 3
        assert "doc_001" in selected_ids
        assert "doc_002" in selected_ids
        assert "doc_003" in selected_ids
        assert isinstance(think, str)
    
    def test_parse_response_with_invalid_ids(self, llm_chooser, sample_summaries):
        """Test parsing LLM response with invalid file IDs"""
        # Simulate LLM response with invalid IDs
        response = "invalid_id_1, invalid_id_2, doc_001"
        
        selected_ids, think = llm_chooser._parse_response(response, sample_summaries)
        
        # Should only return valid IDs
        assert isinstance(selected_ids, list)
        assert "doc_001" in selected_ids
        assert "invalid_id_1" not in selected_ids
        assert "invalid_id_2" not in selected_ids
        assert isinstance(think, str)
    
    def test_parse_response_with_newlines(self, llm_chooser, sample_summaries):
        """Test parsing LLM response with newline-separated IDs"""
        response = "doc_001\ndoc_002\ndoc_003"
        
        selected_ids, think = llm_chooser._parse_response(response, sample_summaries)
        
        assert isinstance(selected_ids, list)
        assert len(selected_ids) == 3
        assert isinstance(think, str)
    
    def test_parse_response_empty_response(self, llm_chooser, sample_summaries):
        """Test parsing empty LLM response"""
        response = ""
        
        selected_ids, think = llm_chooser._parse_response(response, sample_summaries)
        
        assert isinstance(selected_ids, list)
        assert len(selected_ids) == 0
        assert think == ""
    
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