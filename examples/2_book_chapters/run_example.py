"""
Ejemplo 2: Búsqueda en Libro con Capítulos y Metadatos
========================================================

Este ejemplo demuestra el uso de la estrategia WITH_SELECTION_AND_METADATA
(document_selector_metadata_search) para búsquedas precisas en libros.

Escenario:
- 1 libro con múltiples capítulos (book_sample.pdf)

El sistema:
1. Selecciona el documento relevante (el libro)
2. Extrae metadatos de la query (capítulos, páginas, imágenes)
3. Construye filtros precisos de Milvus
4. Busca solo en las secciones específicas solicitadas
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
    """Configura el pipeline de búsqueda con selección y metadatos"""
    
    # Crear modelo LLM para selección y extracción de metadatos
    text_model = OpenAITextModel(model="gpt-4o-mini")
    
    # Configurar pipeline con estrategia WITH_SELECTION_AND_METADATA
    config = SearchPipelineConfig(
        search_type=SearchType.WITH_SELECTION_AND_METADATA,
        collection_name_documents="documents",
        collection_name_summaries="summaries",
        text_model=text_model,
        search_limit=10,
        chooser_max_tokens=500,
        chooser_temperature=0.2
    )
    
    return SearchPipeline(config=config)


def run_book_queries():
    """Ejecuta consultas de ejemplo sobre el libro"""
    
    print("=" * 80)
    print("EJEMPLO 2: BÚSQUEDA EN LIBRO CON CAPÍTULOS Y METADATOS")
    print("=" * 80)
    print("\nDocumentos en el sistema:")
    print("  - book_sample.pdf: Libro con múltiples capítulos")
    print("\n" + "=" * 80 + "\n")
    
    # Queries de ejemplo que incluyen metadatos específicos
    queries = [
        "¿Qué dice el capítulo 3 sobre metodologías de desarrollo?",
        "Busca información en las páginas 10 a 20 sobre arquitectura de software",
        "¿Qué conceptos se explican en el capítulo 1?",
        "Muéstrame información del capítulo 5 sobre testing y pruebas",
        "¿Qué imágenes hay en el capítulo 2?",
        "Busca en las páginas 50-60 información sobre deployment",
        "¿Cuál es el contenido principal del capítulo 7?"
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
                # 1. Selecciona el documento (libro)
                # 2. Extrae metadatos (capítulos, páginas) de la query
                # 3. Construye filtros de Milvus
                # 4. Busca solo en las secciones específicas
                results = pipeline.search(
                    query_embedding=query_embedding,
                    user_query=query  # Requerido para WITH_SELECTION_AND_METADATA
                )
                
                # Mostrar resultados
                if results:
                    print(f"\n✓ Encontrados {len(results)} resultados:\n")
                    for j, result in enumerate(results[:3], 1):  # Mostrar top 3
                        print(f"{j}. Documento: {result.get('file_name', 'N/A')}")
                        print(f"   Score: {result.get('score', 0):.4f}")
                        print(f"   Páginas: {result.get('pages', 'N/A')}")
                        print(f"   Capítulos: {result.get('chapters', 'N/A')}")
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
    
    print("\n📚 Iniciando ejemplo de búsqueda en libro con capítulos...\n")
    
    # Verificar que existe el archivo de datos
    data_dir = Path(__file__).parent / "data"
    book_file = data_dir / "book_sample.pdf"
    
    if not book_file.exists():
        print("⚠️  ADVERTENCIA: No se encontró el archivo 'book_sample.pdf'")
        print("\nPor favor, añade el libro a la carpeta 'data/' antes de ejecutar.")
        print("El archivo debe estar previamente indexado en Milvus con metadatos")
        print("de capítulos y páginas.\n")
        
        response = input("¿Deseas continuar de todas formas? (s/n): ")
        if response.lower() != 's':
            print("Ejemplo cancelado.")
            return
    
    try:
        run_book_queries()
    except Exception as e:
        logger.error(f"Error ejecutando el ejemplo: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        print("\nAsegúrate de que:")
        print("  1. Milvus está corriendo (docker-compose up -d)")
        print("  2. El libro está indexado con metadatos de capítulos y páginas")
        print("  3. Las variables de entorno están configuradas (.env)")
        print("  4. Tienes una API key válida de OpenAI")
        print("\nNota: Este ejemplo requiere que el documento tenga metadatos")
        print("      estructurados (chapters, pages) en Milvus.")


if __name__ == "__main__":
    main()

