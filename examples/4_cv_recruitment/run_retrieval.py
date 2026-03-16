"""
Ejemplo 1: Sistema de Reclutamiento con CVs
============================================

Este ejemplo demuestra el uso de la estrategia WITH_SELECTION (document_selector_search)
para un caso de uso de reclutamiento.

Escenario:
- 1 propuesta de trabajo (job_proposal.pdf)
- 4 CVs de candidatos (cv_candidate_1.pdf, cv_candidate_2.pdf, cv_candidate_3.pdf, cv_candidate_4.pdf)

El sistema primero selecciona qué documentos (CVs) son relevantes para la pregunta
y luego busca información específica dentro de ellos.
"""

import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path para imports
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.llms.text import OpenAITextModel
from src.retrieval.config import SearchPipelineConfig, SearchType
from src.retrieval.pipeline import SearchPipeline
from src.utils import get_logger

logger = get_logger(__name__)

JOB_ID = "retrieval_example_4"

def setup_pipeline():
    """Configura el pipeline de búsqueda con selección de documentos"""
    
    text_model = OpenAITextModel(model="gpt-4o-mini")
    config = SearchPipelineConfig(
        search_type=SearchType.WITH_SELECTION,
        collection_name="cv_recruitment",
        text_model=text_model,
        search_limit=10,
        chooser_max_tokens=500,
        chooser_temperature=0.2,
    )
    
    return SearchPipeline(config=config)


def run_search_queries():
    """Ejecuta consultas de ejemplo sobre los CVs y la propuesta de trabajo"""
    
    print("=" * 80)
    print("EJEMPLO 1: SISTEMA DE RECLUTAMIENTO CON CVs")
    print("=" * 80)
    print("\nDocumentos en el sistema:")
    print("  - job_proposal.pdf: Propuesta de trabajo con requisitos")
    print("  - cv_candidate_1.pdf: CV del Candidato 1")
    print("  - cv_candidate_2.pdf: CV del Candidato 2")
    print("  - cv_candidate_3.pdf: CV del Candidato 3")
    print("  - cv_candidate_4.pdf: CV del Candidato 4")
    print("\n" + "=" * 80 + "\n")
    
    # Queries de ejemplo
    queries = [
        # "¿Quién cumple mejor con los requisitos técnicos de la propuesta de trabajo?",
        "Evalua absolutamente todos los candidatos teniendo en cuenta los requisitos técnicos de la propuesta de trabajo y ponles una nota de 0 a 10, y luego ordénalos de mayor a menor nota."
    ]
    
    # Crear pipeline (cada estrategia genera el embedding de la query internamente)
    with setup_pipeline() as pipeline:
        for i, query in enumerate(queries, 1):
            print(f"\n{'-' * 80}")
            print(f"CONSULTA {i}: {query}")
            print("-" * 80)
            
            try:
                # Realizar búsqueda (la estrategia genera el embedding internamente)
                results = pipeline.search(query=query, job_id=f"{JOB_ID}.{i}")
                
                # Mostrar resultados
                if results:
                    print(f"\n✓ Encontrados {len(results)} resultados:\n")
                    for j, result in enumerate(results[:3], 1):  # Mostrar top 3
                        print(f"{j}. Documento: {result.get('file_name', 'N/A')}")
                        print(f"   Score: {result.get('score', 0):.4f}")
                        print(f"   Páginas: {result.get('pages', 'N/A')}")
                        print(f"   Texto: {result.get('text', '')[:200]}...")
                        print()
                else:
                    print("\n✗ No se encontraron resultados")
                    
            except Exception as e:
                logger.error(f"Error en consulta {i}: {str(e)}", exc_info=True)
                print(f"\n✗ Error: {str(e)}")
    
    print("\n" + "=" * 80)
    print("FIN DE LA BÚSQUEDA")
    print("=" * 80)


def main():
    """Función principal"""
    
    print("\n🚀 Iniciando ejemplo de reclutamiento con CVs...\n")
    
    try:
        run_search_queries()
    except Exception as e:
        logger.error(f"Error ejecutando el ejemplo: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        print("\nAsegúrate de que:")
        print("  1. Milvus está corriendo (docker-compose up -d)")
        print("  2. Los documentos están indexados en las colecciones. (Puedes usar upload_documents.py para indexarlos)")
        print("  3. Las variables de entorno están configuradas (.env)")
        print("  4. Tienes una API key válida de OpenAI")


if __name__ == "__main__":
    main()