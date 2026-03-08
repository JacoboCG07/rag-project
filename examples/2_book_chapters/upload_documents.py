"""
Script de Subida de Documentos - Ejemplo 2: Libro con Capítulos
=================================================================

Este script indexa el libro del ejemplo en Milvus usando el RAG Pipeline.

Documento a subir:
- book_sample.pdf
"""

import sys
from pathlib import Path

# Configurar codificación UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Añadir el directorio raíz al path para imports
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.ingestion.config import IngestionPipelineConfig
from src.utils import get_logger

logger = get_logger(__name__)

JOB_ID = "example_2"


def upload_documents():
    """Sube e indexa el libro en Milvus"""
    
    print("=" * 80)
    print("SUBIDA DE DOCUMENTOS - EJEMPLO 2: LIBRO CON CAPÍTULOS")
    print("=" * 80)
    
    # Ruta a la carpeta de datos
    data_dir = Path(__file__).parent / "data"
    book_file = data_dir / "book_sample.pdf"
    
    # Verificar que existe el archivo
    if not book_file.exists():
        print(f"\n✗ book_sample.pdf - NO ENCONTRADO")
        print("\n❌ No se encontró el archivo 'book_sample.pdf'")
        print("   Por favor, añade el libro a la carpeta 'data/'")
        return
    
    print(f"\n🚀 Procesando 1 documento...\n")
    
    collection_name = "book_chapters"
    
    try:
        config = IngestionPipelineConfig(
            collection_name=collection_name,
            chunk_size=2000,
            extract_images=False,
        )
        
        with IngestionPipeline(config=config) as pipeline:
            print(f"{'─' * 80}")
            print(f"[1/1] Procesando: {book_file.name}")
            print(f"Colección: {collection_name}")
            print(f"  - Partición documentos: documents")
            print(f"  - Partición resúmenes: summaries")
            print('─' * 80)
            
            success, message, result_info = pipeline.process_single_file(
                file_path=str(book_file),
                extract_process_images=False,
                job_id=JOB_ID
            )
            
            if success:
                print(f"\n✓ {book_file.name} procesado correctamente")
                print(f"  - File ID: {result_info.get('file_id', 'N/A')}")
                print(f"  - Mensaje: {message}")
                print("\n" + "=" * 80)
                print("RESUMEN DE SUBIDA")
                print("=" * 80)
                print(f"✓ Exitosos: 1")
                print(f"✗ Fallidos: 0")
                print(f"📊 Total procesados: 1")
                print("\n✅ El libro está listo para búsqueda.")
                print(f"   Colección utilizada: {collection_name}")
                print(f"   - Partición documentos: documents")
                print(f"   - Partición resúmenes: summaries")
                print("   Ejecuta 'python run_example.py' para probar búsquedas por capítulos y páginas.")
            else:
                print(f"\n✗ Error procesando {book_file.name}: {message}")
            
    except Exception as e:
        logger.error(f"Error procesando el libro: {str(e)}", exc_info=True)
        print(f"\n❌ Error crítico: {str(e)}")
        print("\nAsegúrate de que:")
        print("  1. Milvus está corriendo (docker-compose up -d)")
        print("  2. Las variables de entorno están configuradas (.env)")
        print("  3. Tienes una API key válida de OpenAI")


def main():
    """Función principal"""
    
    print("\n📤 Iniciando subida de documento para Ejemplo 2: Libro...\n")
    
    print("ℹ️  Asegúrate de que Milvus está corriendo:")
    print("   docker-compose up -d\n")
    
    upload_documents()


if __name__ == "__main__":
    main()
