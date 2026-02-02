"""
Ejemplo 3: Búsqueda Simple en Manual Técnico
=============================================

Este ejemplo demuestra el uso de la estrategia SIMPLE (simple_search)
para búsquedas directas en documentación técnica.

Escenario:
- 1 manual técnico o documentación (manual.pdf)

El sistema realiza búsqueda vectorial directa sin selección previa de documentos.
Ideal para cuando tienes un solo documento o cuando no necesitas filtrado inteligente.
"""

import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path para imports
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.search.config import SearchPipelineConfig, SearchType
from src.search.pipeline import SearchPipeline
from src.llms.embeddings.openai_embedder import OpenAIEmbedder
from src.utils import get_logger

logger = get_logger(__name__)


def setup_pipeline():
    """Configura el pipeline de búsqueda simple"""
    
    # Configurar pipeline con estrategia SIMPLE
    # No requiere text_model porque no hay selección de documentos
    config = SearchPipelineConfig(
        search_type=SearchType.SIMPLE,
        collection_name_documents="documents",
        search_limit=10
    )
    
    return SearchPipeline(config=config)


def run_manual_queries():
    """Ejecuta consultas de ejemplo sobre el manual técnico"""
    
    print("=" * 80)
    print("EJEMPLO 3: BÚSQUEDA SIMPLE EN MANUAL TÉCNICO")
    print("=" * 80)
    print("\nDocumentos en el sistema:")
    print("  - manual.pdf: Manual técnico o documentación")
    print("\n" + "=" * 80 + "\n")
    
    # Queries de ejemplo típicas de documentación técnica
    queries = [
        "¿Cómo instalar el sistema?",
        "¿Cuáles son los requisitos del sistema?",
        "Explica la configuración inicial",
        "¿Cómo configurar las variables de entorno?",
        "Guía de inicio rápido",
        "¿Cómo solucionar errores comunes?",
        "¿Qué puertos necesito abrir?",
        "Documentación de la API REST",
        "¿Cómo hacer backup de la base de datos?",
        "Procedimiento de actualización del sistema"
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
                
                # Realizar búsqueda simple
                # El pipeline realiza búsqueda vectorial directa en Milvus
                # sin selección previa ni filtros adicionales
                results = pipeline.search(
                    query_embedding=query_embedding
                    # user_query no es necesario para SIMPLE
                    # partition_names y filter_expr son opcionales
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


def run_manual_queries_with_filters():
    """
    Ejemplo adicional: Búsqueda simple con filtros manuales
    
    Aunque la estrategia SIMPLE no usa LLM para selección,
    puedes proporcionar filtros manualmente si conoces el file_id
    o quieres buscar en particiones específicas.
    """
    
    print("\n" + "=" * 80)
    print("EJEMPLO 3B: BÚSQUEDA SIMPLE CON FILTROS MANUALES")
    print("=" * 80)
    
    embedder = OpenAIEmbedder(model="text-embedding-ada-002")
    
    with setup_pipeline() as pipeline:
        query = "¿Cómo configurar el sistema?"
        print(f"\nConsulta: {query}")
        
        query_embedding, _ = embedder.generate_embedding(text=query)
        
        # Ejemplo 1: Búsqueda con filtro de tipo de archivo
        print("\n1. Búsqueda solo en PDFs:")
        results_pdf = pipeline.search(
            query_embedding=query_embedding,
            filter_expr='type_file == "PDF"'
        )
        print(f"   Resultados: {len(results_pdf)}")
        
        # Ejemplo 2: Búsqueda en un documento específico (si conoces el file_id)
        print("\n2. Búsqueda en documento específico:")
        print("   (Requiere conocer el file_id del manual)")
        # results_specific = pipeline.search(
        #     query_embedding=query_embedding,
        #     filter_expr='file_id == "manual_123"'
        # )
        
        # Ejemplo 3: Búsqueda en particiones específicas
        print("\n3. Búsqueda en particiones específicas:")
        print("   (Si tu colección usa particiones)")
        # results_partition = pipeline.search(
        #     query_embedding=query_embedding,
        #     partition_names=["technical_docs"]
        # )
    
    print("\n" + "=" * 80)


def main():
    """Función principal"""
    
    print("\n📖 Iniciando ejemplo de búsqueda simple en manual técnico...\n")
    
    # Verificar que existe el archivo de datos
    data_dir = Path(__file__).parent / "data"
    manual_file = data_dir / "manual.pdf"
    
    if not manual_file.exists():
        print("⚠️  ADVERTENCIA: No se encontró el archivo 'manual.pdf'")
        print("\nPor favor, añade el manual a la carpeta 'data/' antes de ejecutar.")
        print("El archivo debe estar previamente indexado en Milvus.\n")
        
        response = input("¿Deseas continuar de todas formas? (s/n): ")
        if response.lower() != 's':
            print("Ejemplo cancelado.")
            return
    
    try:
        # Ejecutar búsquedas simples
        run_manual_queries()
        
        # Ejecutar ejemplo con filtros (opcional)
        print("\n¿Deseas ver el ejemplo con filtros manuales? (s/n): ", end="")
        if input().lower() == 's':
            run_manual_queries_with_filters()
            
    except Exception as e:
        logger.error(f"Error ejecutando el ejemplo: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        print("\nAsegúrate de que:")
        print("  1. Milvus está corriendo (docker-compose up -d)")
        print("  2. El manual está indexado en la colección 'documents'")
        print("  3. Las variables de entorno están configuradas (.env)")
        print("  4. Tienes una API key válida de OpenAI (para embeddings)")
        print("\nNota: Esta estrategia NO requiere LLM para búsqueda,")
        print("      solo para generar embeddings de las queries.")


if __name__ == "__main__":
    main()





