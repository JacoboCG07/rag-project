"""
Ejemplo 2: Chatbot RAG — Reclutamiento con CVs (vía API chatbot)
================================================================

Llama al servicio HTTP de chat (POST /chat): retrieval con `with_selection` + LLM.

Requisito: servicio chatbot accesible (por defecto http://localhost:8018).
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.utils import get_logger

logger = get_logger(__name__)

CHATBOT_BASE_URL = "http://localhost:8018"

MODEL = "gpt-4o-mini"
SEARCH_TYPE = "with_selection"
SEARCH_LIMIT = 10

COLLECTION_NAME = "cv_recruitment"
JOB_ID = "chatbot_example_2"


def main() -> None:
    print("\n" + "=" * 80)
    print("CHATBOT RAG (API chatbot) - Sistema de Reclutamiento con CVs")
    print("=" * 80)
    print(f"Servicio: {CHATBOT_BASE_URL}")
    print(f"Colección en POST /chat: {COLLECTION_NAME}")
    print(f"Estrategia: {SEARCH_TYPE}")
    print("=" * 80 + "\n")

    try:
        r = httpx.get(f"{CHATBOT_BASE_URL}/health", timeout=10.0)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ No se puede contactar con el servicio chatbot: {e}")
        return

    queries = [
        "Evalúa los mejores candidatos teniendo en cuenta los requisitos técnicos "
        "de la propuesta de trabajo y ponles una nota de 0 a 10, y luego ordénalos de mayor a menor nota.",
    ]

    with httpx.Client(timeout=180.0) as client:
        for i, query in enumerate(queries, 1):
            print(f"\n{'-' * 80}")
            print(f"PREGUNTA {i}: {query}")
            print("-" * 80)
            try:
                payload = {
                    "query": query,
                    "collection_name": COLLECTION_NAME,
                    "search_type": SEARCH_TYPE,
                    "search_limit": SEARCH_LIMIT,
                    "llm_model": MODEL,
                    "job_id": f"{JOB_ID}.{i}",
                }
                resp = client.post(f"{CHATBOT_BASE_URL}/chat", json=payload)
                if resp.status_code != 200:
                    print(f"\nHTTP {resp.status_code}: {resp.text}")
                    continue
                data = resp.json()
                answer = data.get("answer", "")
                print(f"\nRespuesta:\n{answer}\n")
            except Exception as e:
                logger.error("Error en pregunta %s: %s", i, e, exc_info=True)
                print(f"\nError: {e}\n")

    print("=" * 80)
    print("FIN")
    print("=" * 80)


if __name__ == "__main__":
    main()