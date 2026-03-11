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

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.llms.text import OpenAITextModel
from src.retrieval.config import SearchPipelineConfig, SearchType
from src.retrieval.pipeline import SearchPipeline
from src.utils import get_logger

logger = get_logger(__name__)


def setup_pipeline():
    """Configura el pipeline de búsqueda con selección y metadatos"""
    text_model = OpenAITextModel(model="gpt-4o-mini")
    config = SearchPipelineConfig(
        search_type=SearchType.WITH_SELECTION_AND_METADATA,
        collection_name="book_chapters",
        text_model=text_model,
        search_limit=10,
        chooser_max_tokens=500,
        chooser_temperature=0.2,
    )
    return SearchPipeline(config=config)


def run_search_queries():
    """Ejecuta consultas de ejemplo sobre el libro"""
    print("=" * 80)
    print("EJEMPLO 2: BÚSQUEDA EN LIBRO CON CAPÍTULOS Y METADATOS")
    print("=" * 80)
    print("\nDocumentos en el sistema:")
    print("  - book_sample.pdf: Libro con múltiples capítulos")
    print("\n" + "=" * 80 + "\n")
    
    queries = [
        "¿Qué dice el capítulo 3 sobre metodologías de desarrollo?"
    ]
    
    with setup_pipeline() as pipeline:
        for i, query in enumerate(queries, 1):
            print(f"\n{'-' * 80}")
            print(f"CONSULTA {i}: {query}")
            print("-" * 80)
            try:
                results = pipeline.search(query=query)
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
                logger.error(f"Error en consulta {i}: {str(e)}", exc_info=True)
                print(f"\n✗ Error: {str(e)}")
    
    print("\n" + "=" * 80)
    print("FIN DE LA BÚSQUEDA")
    print("=" * 80)


def main():
    """Función principal"""
    print("\n📚 Iniciando ejemplo de búsqueda en libro con capítulos...\n")
    
    data_dir = Path(__file__).parent / "data"
    book_file = data_dir / "book_sample.pdf"
    
    if not book_file.exists():
        print("⚠️  ADVERTENCIA: No se encontró el archivo 'book_sample.pdf'")
        print("\nAñade el libro a la carpeta 'data/' y ejecuta antes 'python upload_documents.py'.\n")
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
        print("  2. El libro está indexado (python upload_documents.py)")
        print("  3. Las variables de entorno están configuradas (.env)")
        print("  4. Tienes una API key válida de OpenAI")


if __name__ == "__main__":
    main()