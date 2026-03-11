"""
Script de Subida de Documentos - Ejemplo 2: Libro con Capítulos
=================================================================

Este script indexa el libro del ejemplo en Milvus usando el RAG Pipeline.
La búsqueda en este ejemplo usa metadatos (capítulos, páginas) con WITH_SELECTION_AND_METADATA.

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
COLLECTION_NAME = "book_chapters"


def upload_documents():
    """Sube e indexa el libro en Milvus"""
    
    print("=" * 80)
    print("SUBIDA DE DOCUMENTOS - EJEMPLO 2: LIBRO CON CAPÍTULOS")
    print("=" * 80)
    
    data_dir = Path(__file__).parent / "data"
    expected_files = ["book_sample.pdf"]
    existing_files = [data_dir / f for f in expected_files]
    
    # Verificar que existen los archivos
    missing = [f.name for f in existing_files if not f.exists()]
    if missing:
        print(f"\n✗ Archivo(s) no encontrado(s): {', '.join(missing)}")
        print("   Añade el libro a la carpeta 'data/'")
        return
    
    print(f"\n🚀 Procesando {len(existing_files)} documento(s)...\n")
    
    try:
        config = IngestionPipelineConfig(
            collection_name=COLLECTION_NAME,
            chunk_size=2000,
            extract_images=False,
        )
        
        with IngestionPipeline(config=config) as pipeline:
            successful = 0
            failed = 0
            
            for i, file_path in enumerate(existing_files, 1):
                print(f"\n{'─' * 80}")
                print(f"[{i}/{len(existing_files)}] Procesando: {file_path.name}")
                print(f"Colección: {COLLECTION_NAME}")
                print(f"  - Partición documentos: documents")
                print(f"  - Partición resúmenes: summaries")
                print('─' * 80)
                
                try:
                    success, message, result_info = pipeline.process_single_file(
                        file_path=str(file_path),
                        job_id=JOB_ID
                    )
                    if success:
                        print(f"✓ {file_path.name} procesado correctamente")
                        print(f"  - File ID: {result_info.get('file_id', 'N/A')}")
                        print(f"  - Mensaje: {message}")
                        successful += 1
                    else:
                        print(f"✗ Error procesando {file_path.name}: {message}")
                        failed += 1
                except Exception as e:
                    logger.error(f"Error procesando {file_path.name}: {str(e)}", exc_info=True)
                    print(f"✗ Error procesando {file_path.name}: {str(e)}")
                    failed += 1
            
            print("\n" + "=" * 80)
            print("RESUMEN DE SUBIDA")
            print("=" * 80)
            print(f"✓ Exitosos: {successful}")
            print(f"✗ Fallidos: {failed}")
            print(f"📊 Total procesados: {successful + failed}")
            
            if successful > 0:
                print("\n✅ El libro está listo para búsqueda con metadatos (capítulos, páginas).")
                print(f"   Colección utilizada: {COLLECTION_NAME}")
                print(f"   - Partición documentos: documents")
                print(f"   - Partición resúmenes: summaries")
                print("   Ejecuta 'python run_retrieval.py' para probar búsquedas por capítulos/páginas.")
                print("   Ejecuta 'python run_chatbot.py' para el chatbot con filtros por metadatos.")
            
    except Exception as e:
        logger.error(f"Error en el pipeline: {str(e)}", exc_info=True)
        print(f"\n❌ Error crítico: {str(e)}")
        print("\nAsegúrate de que:")
        print("  1. Milvus está corriendo (docker-compose up -d)")
        print("  2. Las variables de entorno están configuradas (.env)")
        print("  3. Tienes una API key válida de OpenAI")


def main():
    """Función principal"""
    print("\n📤 Iniciando subida de documento para Ejemplo 2: Libro con capítulos...\n")
    print("ℹ️  Asegúrate de que Milvus está corriendo:")
    print("   docker-compose up -d\n")
    upload_documents()


if __name__ == "__main__":
    main()
