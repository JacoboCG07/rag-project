"""
Prueba 1: Búsqueda simple sobre El Principito
=============================================

Este ejemplo demuestra la búsqueda SIMPLE (vector search directo en Milvus)
sobre documentos indexados.

Escenario:
- principito.pdf indexado en la colección principito_imagenes

El sistema busca fragmentos similares a la consulta usando embeddings.
"""

import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path para imports
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.retrieval.config import SearchPipelineConfig, SearchType
from src.retrieval.pipeline import SearchPipeline
from src.utils import get_logger

logger = get_logger(__name__)


def setup_pipeline():
    """Configura el pipeline de búsqueda simple"""
    
    # Configurar pipeline con estrategia SIMPLE
    # Usa la misma colección que en upload_documents.py
    config = SearchPipelineConfig(
        search_type=SearchType.SIMPLE,
        collection_name="principito_imagenes",  # Misma colección que en upload_documents.py
        search_limit=10,
    )
    
    return SearchPipeline(config=config)


def run_search_queries():
    """Ejecuta consultas de ejemplo sobre El Principito"""
    
    print("=" * 80)
    print("PRUEBA 1: BÚSQUEDA SIMPLE - El Principito")
    print("=" * 80)
    print("\nDocumentos en el sistema:")
    print("  - principito.pdf")
    print("\n" + "=" * 80 + "\n")
    
    # Queries de ejemplo
    queries = [
        "¿Qué le pidió el principito al aviador que dibujara?",
        "¿Dónde aterrizó el aviador?",
        "¿Qué significa domesticar?",
        "¿Cómo es el planeta del principito?",
        "¿Qué hace el zorro con el principito?"
    ]
    
    # Crear pipeline (cada estrategia genera el embedding de la query internamente)
    with setup_pipeline() as pipeline:
        for i, query in enumerate(queries, 1):
            print(f"\n{'─' * 80}")
            print(f"CONSULTA {i}: {query}")
            print('─' * 80)
            
            try:
                # Realizar búsqueda (la estrategia genera el embedding internamente)
                results = pipeline.search(query=query)
                
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
    
    print("\n🚀 Iniciando búsqueda simple sobre El Principito...\n")
    
    # Verificar que existen los archivos de datos (referencia)
    data_dir = Path(__file__).parent / "data"
    expected_files = ["principito.pdf"]
    
    missing_files = [f for f in expected_files if not (data_dir / f).exists()]
    
    if missing_files:
        print("⚠️  ADVERTENCIA: Los siguientes archivos no se encontraron en data/:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nEjecuta primero 'python upload_documents.py' para indexar los documentos.\n")
        
        response = input("¿Deseas continuar de todas formas? (s/n): ")
        if response.lower() != 's':
            print("Ejemplo cancelado.")
            return
    
    try:
        run_search_queries()
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