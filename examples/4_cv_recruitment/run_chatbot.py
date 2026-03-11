"""
Ejemplo 4: Chatbot RAG - Sistema de Reclutamiento con CVs
==========================================================

Flujo completo RAG: retrieval con selección de documentos + generación de respuestas con LLM.
Usa la misma colección que upload_documents.py y run_retrieval.py: cv_recruitment.
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

JOB_ID = "chatbot_example_4"


def create_chatbot_config() -> ChatbotConfig:
    """Configuración del chatbot (retriever con selección + LLM)."""
    text_model = OpenAITextModel(model="gpt-4o-mini")
    return ChatbotConfig(
        retrieval=SearchPipelineConfig(
            search_type=SearchType.WITH_SELECTION,
            collection_name="cv_recruitment",
            text_model=text_model,
            search_limit=10,
            chooser_max_tokens=500,
            chooser_temperature=0.2,
        ),
        llm_model="gpt-4o-mini",
    )


def main():
    print("\n" + "=" * 80)
    print("CHATBOT RAG - Sistema de Reclutamiento con CVs")
    print("=" * 80)
    print("\nDocumentos: job_proposal.pdf + cv_candidate_1..4.pdf (colección cv_recruitment)")
    print("=" * 80 + "\n")

    config = create_chatbot_config()
    queries = [
        # "¿Quién cumple mejor con los requisitos técnicos de la propuesta de trabajo?",
        # "¿Qué candidato tiene más experiencia en liderazgo?",
        # "Resume los requisitos del puesto y qué candidatos los cubren.",
        "Pon una nota a todos los candidatos según los requisitos del puesto y la propuesta de trabajo."
    ]

    with RAGChatbot(config=config) as bot:
        for i, query in enumerate(queries, 1):
            print(f"\n{'-' * 80}")
            print(f"PREGUNTA {i}: {query}")
            print("-" * 80)
            try:
                answer = bot.ask(query, job_id=f"{JOB_ID}.{i}")
                print(f"\nRespuesta:\n{answer}\n")
            except Exception as e:
                logger.error(f"Error en pregunta {i}: {str(e)}", exc_info=True)
                print(f"\nError: {str(e)}\n")

    print("=" * 80)
    print("FIN")
    print("=" * 80)


if __name__ == "__main__":
    main()
