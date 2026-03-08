"""
Script de Subida de Documentos - Ejemplo 3: Manual Técnico
============================================================

Este script indexa el manual técnico del ejemplo en Milvus usando el RAG Pipeline.

Documento a subir:
- manual.pdf
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

JOB_ID = "example_4"
EXTRACT_IMAGES = True
MAX_IMAGES_PER_DOCUMENT = 100

def upload_documents():
    """Sube e indexa el manual técnico en Milvus"""
    
    print("=" * 80)
    print("SUBIDA DE DOCUMENTOS - EJEMPLO 3: MANUAL TÉCNICO")
    print("=" * 80)
    
    # Ruta a la carpeta de datos
    data_dir = Path(__file__).parent / "data"
    manual_file = data_dir / "social_media_report.pdf"
    
    # Verificar que existe el archivo
    if not manual_file.exists():
        print(f"\n✗ {manual_file.name} - NO ENCONTRADO")
        print("\n❌ No se encontró el archivo 'manual.pdf'")
        print("   Por favor, añade el manual a la carpeta 'data/'")
        return
    
    print(f"\n🚀 Procesando 1 documento...\n")
    
    collection_name = "technical_manual"
    
    try:
        config = IngestionPipelineConfig(
            collection_name=collection_name,
            chunk_size=2000,
            extract_images=EXTRACT_IMAGES,
            max_images_per_document=MAX_IMAGES_PER_DOCUMENT,
        )
        
        with IngestionPipeline(config=config) as pipeline:
            print(f"{'─' * 80}")
            print(f"[1/1] Procesando: {manual_file.name}")
            print(f"Colección: {collection_name}")
            print(f"  - Partición documentos: documents")
            print(f"  - Partición resúmenes: summaries")
            print('─' * 80)
            
            success, message, result_info = pipeline.process_single_file(
                file_path=str(manual_file),
                job_id=JOB_ID
            )
            
            if success:
                print(f"\n✓ {manual_file.name} procesado correctamente")
                print(f"  - File ID: {result_info.get('file_id', 'N/A')}")
                print(f"  - Mensaje: {message}")
                print("\n" + "=" * 80)
                print("RESUMEN DE SUBIDA")
                print("=" * 80)
                print(f"✓ Exitosos: 1")
                print(f"✗ Fallidos: 0")
                print(f"📊 Total procesados: 1")
                print("\n✅ El manual está listo para búsqueda.")
                print(f"   Colección utilizada: {collection_name}")
                print(f"   - Partición documentos: documents")
                print(f"   - Partición resúmenes: summaries")
                print("   Ejecuta 'python run_example.py' para probar búsquedas directas.")
            else:
                print(f"\n✗ Error procesando {manual_file.name}: {message}")
            
    except Exception as e:
        logger.error(f"Error procesando el manual: {str(e)}", exc_info=True)
        print(f"\n❌ Error crítico: {str(e)}")
        print("\nAsegúrate de que:")
        print("  1. Milvus está corriendo (docker-compose up -d)")
        print("  2. Las variables de entorno están configuradas (.env)")
        print("  3. Tienes una API key válida de OpenAI")


def main():
    """Función principal"""
    
    print("\n📤 Iniciando subida de documento para Ejemplo 3: Manual Técnico...\n")
    
    print("ℹ️  Asegúrate de que Milvus está corriendo:")
    print("   docker-compose up -d\n")
    
    upload_documents()


if __name__ == "__main__":
    main()