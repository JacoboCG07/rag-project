"""
Prueba 1: Búsqueda con selección y metadatos sobre El Principito (vía API retrieval)
===================================================================================

Llama al servicio HTTP de búsqueda (POST /search) con `search_type: with_selection_and_metadata`.

Requisito: servicio retrieval accesible (por defecto http://localhost:8007).
"""

from __future__ import annotations
import sys
from pathlib import Path
import httpx

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.utils import get_logger

logger = get_logger(__name__)

RETRIEVAL_BASE_URL = "http://localhost:8007"
SEARCH_TYPE = "with_selection_and_metadata"
SEARCH_LIMIT = 10

COLLECTION_NAME = "principito_imagenes"
JOB_ID = "retrieval_example_1"

def run_search_queries() -> None:
    print("=" * 80)
    print("PRUEBA 1: WITH_SELECTION_AND_METADATA (API retrieval) - El Principito")
    print("=" * 80)
    print(f"Servicio: {RETRIEVAL_BASE_URL}")
    print("\n" + "=" * 80 + "\n")

    queries = [
        "¿Hazme un resumen del capítulo 1 y 2?",
    ]

    try:
        r = httpx.get(f"{RETRIEVAL_BASE_URL}/health", timeout=10.0)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ No se puede contactar con el servicio retrieval: {e}")
        return

    with httpx.Client(timeout=120.0) as client:
        for i, query in enumerate(queries, 1):
            print(f"\n{'─' * 80}")
            print(f"CONSULTA {i}: {query}")
            print("─" * 80)
            try:
                payload = {
                    "query": query,
                    "collection_name": COLLECTION_NAME,
                    "search_type": SEARCH_TYPE,
                    "search_limit": SEARCH_LIMIT,
                    "job_id": f"{JOB_ID}.{i}",
                }
                resp = client.post(f"{RETRIEVAL_BASE_URL}/search", json=payload)
                if resp.status_code != 200:
                    print(f"\n✗ HTTP {resp.status_code}: {resp.text}")
                    continue
                data = resp.json()
                results = data.get("results") or []

                if results:
                    print(f"\n✓ Encontrados {len(results)} resultados:\n")
                    for j, result in enumerate(results[:3], 1):
                        print(f"{j}. Documento: {result.get('file_name', 'N/A')}")
                        print(f"   Score: {result.get('score', 0):.4f}")
                        print(f"   Páginas: {result.get('pages', 'N/A')}")
                        print(f"   Capítulos: {result.get('chapters', 'N/A')}")
                        print(f"   Texto: {result.get('text', '')[:200]}...")
                        print()
                else:
                    print("\n✗ No se encontraron resultados")

            except Exception as e:
                logger.error("Error en consulta %s: %s", i, e, exc_info=True)
                print(f"\n✗ Error: {e}")

    print("\n" + "=" * 80)
    print("FIN DE LA BÚSQUEDA")
    print("=" * 80)


def main() -> None:
    print("\n🚀 Búsqueda vía API retrieval - El Principito\n")
    try:
        run_search_queries()
    except Exception as e:
        logger.error("Error ejecutando el ejemplo: %s", e, exc_info=True)
        print(f"\n❌ Error: {e}")
        print("\nAsegúrate de que el servicio retrieval está levantado y los documentos indexados.")


if __name__ == "__main__":
    main()