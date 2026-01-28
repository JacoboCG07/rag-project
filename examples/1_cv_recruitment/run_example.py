"""
Ejemplo 1: Sistema de Reclutamiento con CVs
============================================

Este ejemplo demuestra el uso de la estrategia WITH_SELECTION (document_selector_search)
para un caso de uso de reclutamiento.

Escenario:
- 1 propuesta de trabajo (job_proposal.pdf)
- 3 CVs de candidatos (cv_candidate_1.pdf, cv_candidate_2.pdf, cv_candidate_3.pdf)

El sistema primero selecciona qué documentos (CVs) son relevantes para la pregunta
y luego busca información específica dentro de ellos.
"""

import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path para imports
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from llms.text import OpenAITextModel
from src.search.config import SearchPipelineConfig, SearchType
from src.search.pipeline import SearchPipeline
from rag.processing.embeddings.openai_embedder import OpenAIEmbedder
from src.utils import get_logger

logger = get_logger(__name__)


def setup_pipeline():
    """Configura el pipeline de búsqueda con selección de documentos"""
    
    # Crear modelo LLM para selección de documentos
    text_model = OpenAITextModel(model="gpt-4o-mini")
    
    # Configurar pipeline con estrategia WITH_SELECTION
    config = SearchPipelineConfig(
        search_type=SearchType.WITH_SELECTION,
        collection_name_documents="documents",
        collection_name_summaries="summaries",
        text_model=text_model,
        search_limit=10,
        chooser_max_tokens=500,
        chooser_temperature=0.2
    )
    
    return SearchPipeline(config=config)


def run_recruitment_queries():
    """Ejecuta consultas de ejemplo sobre los CVs y la propuesta de trabajo"""
    
    print("=" * 80)
    print("EJEMPLO 1: SISTEMA DE RECLUTAMIENTO CON CVs")
    print("=" * 80)
    print("\nDocumentos en el sistema:")
    print("  - job_proposal.pdf: Propuesta de trabajo con requisitos")
    print("  - cv_candidate_1.pdf: CV del Candidato 1")
    print("  - cv_candidate_2.pdf: CV del Candidato 2")
    print("  - cv_candidate_3.pdf: CV del Candidato 3")
    print("\n" + "=" * 80 + "\n")
    
    # Queries de ejemplo
    queries = [
        "¿Qué candidato tiene experiencia en Python y desarrollo backend?",
        "¿Quién cumple mejor con los requisitos técnicos de la propuesta de trabajo?",
        "¿Qué candidato tiene más años de experiencia profesional?",
        "¿Algún candidato tiene experiencia con bases de datos vectoriales o Milvus?",
        "¿Qué formación académica tienen los candidatos?",
        "¿Cuáles son los requisitos principales de la propuesta de trabajo?"
    ]
    
    # Inicializar embedder
    embedder = OpenAIEmbedder(model="text-embedding-ada-002")
    
    # Crear pipeline
    with setup_pipeline() as pipeline:
        for i, query in enumerate(queries, 1):
            print(f"\n{'─' * 80}")
            print(f"CONSULTA {i}: {query}")
            print('─' * 80)
            
            try:
                # Generar embedding de la query
                query_embedding, _ = embedder.generate_embedding(text=query)
                
                # Realizar búsqueda
                # El pipeline automáticamente:
                # 1. Selecciona documentos relevantes usando el LLM
                # 2. Busca en esos documentos seleccionados
                results = pipeline.search(
                    query_embedding=query_embedding,
                    user_query=query  # Requerido para WITH_SELECTION
                )
                
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
    print("FIN DEL EJEMPLO")
    print("=" * 80)


def main():
    """Función principal"""
    
    print("\n🚀 Iniciando ejemplo de reclutamiento con CVs...\n")
    
    # Verificar que existen los archivos de datos
    data_dir = Path(__file__).parent / "data"
    expected_files = [
        "job_proposal.pdf",
        "cv_candidate_1.pdf",
        "cv_candidate_2.pdf",
        "cv_candidate_3.pdf"
    ]
    
    missing_files = [f for f in expected_files if not (data_dir / f).exists()]
    
    if missing_files:
        print("⚠️  ADVERTENCIA: Los siguientes archivos no se encontraron:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPor favor, añade los documentos a la carpeta 'data/' antes de ejecutar.")
        print("Los archivos deben estar previamente indexados en Milvus.\n")
        
        response = input("¿Deseas continuar de todas formas? (s/n): ")
        if response.lower() != 's':
            print("Ejemplo cancelado.")
            return
    
    try:
        run_recruitment_queries()
    except Exception as e:
        logger.error(f"Error ejecutando el ejemplo: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        print("\nAsegúrate de que:")
        print("  1. Milvus está corriendo (docker-compose up -d)")
        print("  2. Los documentos están indexados en las colecciones")
        print("  3. Las variables de entorno están configuradas (.env)")
        print("  4. Tienes una API key válida de OpenAI")


if __name__ == "__main__":
    main()

