"""
Prueba 1: Chatbot RAG sobre El Principito
=========================================

Flujo completo RAG: retrieval + generación de respuestas con LLM.
Usa la misma colección que run_example.py y upload_documents.py: principito_imagenes.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.chatbot import ChatbotConfig, RAGChatbot
from src.retrieval.config import SearchPipelineConfig, SearchType
from src.utils import get_logger

logger = get_logger(__name__)


def create_chatbot_config() -> ChatbotConfig:
    """Configuración del chatbot (retriever + LLM)."""
    return ChatbotConfig(
        retrieval=SearchPipelineConfig(
            search_type=SearchType.SIMPLE,
            collection_name="principito_imagenes",
            search_limit=10,
        ),
        llm_model="gpt-4o-mini",
    )


def main():
    print("\n" + "=" * 80)
    print("CHATBOT RAG - El Principito")
    print("=" * 80)
    print("\nDocumentos: principito.pdf (colección principito_imagenes)")
    print("=" * 80 + "\n")

    config = create_chatbot_config()
    queries = [
        "¿Qué le pidió el principito al aviador que dibujara?",
        "¿Dónde aterrizó el aviador?",
        "¿Qué significa domesticar?",
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
