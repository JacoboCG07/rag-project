"""
Ejemplo 2: Chatbot RAG - Libro con Capítulos y Metadatos
=========================================================

Flujo completo RAG: retrieval con selección y metadatos + generación de respuestas con LLM.
Usa la misma colección que upload_documents.py y run_retrieval.py: book_chapters.
La búsqueda incorpora filtros por capítulos, páginas e imágenes.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.chatbot import ChatbotConfig, RAGChatbot
from src.llms.text import OpenAITextModel
from src.retrieval.config import SearchPipelineConfig, SearchType
from src.utils import get_logger

logger = get_logger(__name__)


def create_chatbot_config() -> ChatbotConfig:
    """Configuración del chatbot (retriever con selección + metadatos + LLM)."""
    text_model = OpenAITextModel(model="gpt-4o-mini")
    return ChatbotConfig(
        retrieval=SearchPipelineConfig(
            search_type=SearchType.WITH_SELECTION_AND_METADATA,
            collection_name="book_chapters",
            text_model=text_model,
            search_limit=10,
            chooser_max_tokens=500,
            chooser_temperature=0.2,
        ),
        llm_model="gpt-4o-mini",
    )


def main():
    print("\n" + "=" * 80)
    print("CHATBOT RAG - Libro con Capítulos y Metadatos")
    print("=" * 80)
    print("\nDocumento: book_sample.pdf (colección book_chapters, búsqueda con metadatos)")
    print("=" * 80 + "\n")

    config = create_chatbot_config()
    queries = [
        "¿Qué dice el capítulo 3 sobre metodologías de desarrollo?",
        "Resume los conceptos principales del capítulo 1.",
        "Busca en las páginas 10 a 20 información sobre arquitectura.",
    ]

    with RAGChatbot(config=config) as bot:
        for i, query in enumerate(queries, 1):
            print(f"\n{'-' * 80}")
            print(f"PREGUNTA {i}: {query}")
            print("-" * 80)
            try:
                answer = bot.ask(query)
                print(f"\nRespuesta:\n{answer}\n")
            except Exception as e:
                logger.error(f"Error en pregunta {i}: {str(e)}", exc_info=True)
                print(f"\nError: {str(e)}\n")

    print("=" * 80)
    print("FIN")
    print("=" * 80)


if __name__ == "__main__":
    main()
