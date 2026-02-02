"""
Script de Subida de Documentos - Ejemplo 2: Libro con Capítulos
=================================================================

Este script indexa el libro del ejemplo en Milvus usando el RAG Pipeline.

Documento a subir:
- book_sample.pdf
"""

import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path para imports
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.rag.rag_pipeline import RAGPipeline
from src.rag.config import RAGPipelineConfig
from src.utils import get_logger

logger = get_logger(__name__)


def upload_documents():
    """Sube e indexa el libro en Milvus"""
    
    print("=" * 80)
    print("SUBIDA DE DOCUMENTOS - EJEMPLO 2: LIBRO CON CAPÍTULOS")
    print("=" * 80)
    
    # Ruta a la carpeta de datos
    data_dir = Path(__file__).parent / "data"
    book_file = data_dir / "book_sample.pdf"
    
    # Verificar que existe el archivo
    print("\n📋 Verificando archivo...")
    
    if not book_file.exists():
        print(f"  ✗ book_sample.pdf - NO ENCONTRADO")
        print("\n❌ No se encontró el archivo 'book_sample.pdf'")
        print("   Por favor, añade el libro a la carpeta 'data/'")
        return
    
    print(f"  ✓ book_sample.pdf")
    print(f"\n🚀 Procesando libro...\n")
    
    try:
        # Configurar el RAG Pipeline
        # Nota: Ajusta la configuración según tus necesidades
        config = RAGPipelineConfig()
        
        with RAGPipeline(config=config) as pipeline:
            print(f"{'─' * 80}")
            print(f"Procesando: {book_file.name}")
            print('─' * 80)
            
            # Procesar e indexar el documento
            result = pipeline.process_document(str(book_file))
            
            print(f"\n✓ {book_file.name} procesado correctamente")
            if result:
                print(f"  - Chunks generados: {result.get('chunks_count', 'N/A')}")
                print(f"  - File ID: {result.get('file_id', 'N/A')}")
                print(f"  - Capítulos detectados: {result.get('chapters', 'N/A')}")
                print(f"  - Páginas totales: {result.get('total_pages', 'N/A')}")
            
            print("\n" + "=" * 80)
            print("✅ LIBRO INDEXADO CORRECTAMENTE")
            print("=" * 80)
            print("\nEl libro está listo para búsqueda con metadatos.")
            print("Ejecuta 'python run_example.py' para probar búsquedas por capítulos y páginas.")
            
    except Exception as e:
        logger.error(f"Error procesando el libro: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        print("\nAsegúrate de que:")
        print("  1. Milvus está corriendo (docker-compose up -d)")
        print("  2. Las variables de entorno están configuradas (.env)")
        print("  3. Tienes una API key válida de OpenAI")
        print("  4. El libro tiene estructura de capítulos y páginas")
        print("\nNota: Este ejemplo requiere que el documento tenga metadatos")
        print("      estructurados (capítulos, páginas) para funcionar correctamente.")


def main():
    """Función principal"""
    
    print("\n📤 Iniciando subida de documento para Ejemplo 2: Libro...\n")
    
    # Verificar que Milvus está disponible
    print("ℹ️  Asegúrate de que Milvus está corriendo:")
    print("   docker-compose up -d\n")
    
    response = input("¿Milvus está corriendo? (s/n): ")
    if response.lower() != 's':
        print("\nPor favor, inicia Milvus primero:")
        print("  cd ../../  # Ir a la raíz del proyecto")
        print("  docker-compose up -d")
        return
    
    upload_documents()


if __name__ == "__main__":
    main()





