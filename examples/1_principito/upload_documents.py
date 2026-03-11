"""
Script de Subida de Documentos - Ejemplo 1: CVs y Propuesta de Trabajo
========================================================================

Este script indexa los documentos del ejemplo en Milvus usando el RAG Pipeline.

Documentos a subir:
- principito.pdf
"""

import os
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

JOB_ID = "prueba_1"
EXTRACT_IMAGES = True

def upload_documents():
    """Sube e indexa todos los documentos del ejemplo en Milvus"""
    
    print("=" * 80)
    print("SUBIDA DE DOCUMENTOS - Pruebas de RAG")
    print("=" * 80)
    
    if EXTRACT_IMAGES:
        print("\n⚠️  EXTRACT_IMAGES está activado (True).")
        print("   La búsqueda incorporará las imágenes de los documentos.")
        print("   Esto conlleva más gasto en OpenAI (por cada imagen se hará una descripción).")
        print("   Si no quieres incorporar las imágenes modifica la variable EXTRACT_IMAGES de este archivo a False.\n")
        while True:
            respuesta = input("¿Quieres continuar? (s/n): ").strip().lower()
            if respuesta in ("s", "si", "y", "yes"):
                break
            if respuesta in ("n", "no"):
                print("Subida cancelada.")
                return
            print("Responde 's' para sí o 'n' para no.")
    
    # Ruta a la carpeta de datos
    data_dir = Path(__file__).parent / "data"
    
    # Lista de documentos a procesar
    expected_files = [
        "principito.pdf",
    ]
    
    # Obtener rutas de los archivos
    existing_files = [data_dir / filename for filename in expected_files]
    print(f"\n🚀 Procesando {len(existing_files)} documento(s)...\n")
    
    # Nombre de la colección para este ejemplo
    # La colección tendrá dos particiones: 'documents' y 'summaries'
    collection_name = "principito_imagenes"
    
    try:
        # Configurar el RAG Pipeline
        # Opciones: chunk_size, chunk_overlap, extract_images
        config = IngestionPipelineConfig(
            collection_name=collection_name,
            chunk_size=2000,
            extract_images=EXTRACT_IMAGES,
        )
        
        with IngestionPipeline(config=config) as pipeline:
            successful = 0
            failed = 0
            
            for i, file_path in enumerate(existing_files, 1):
                print(f"\n{'─' * 80}")
                print(f"[{i}/{len(existing_files)}] Procesando: {file_path.name}")
                print(f"Colección: {collection_name}")
                print(f"  - Partición documentos: documents")
                print(f"  - Partición resúmenes: summaries")
                print('─' * 80)
                
                try:
                    # Procesar e indexar el documento
                    # Los documentos van a la partición 'documents' y los resúmenes a 'summaries'
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
                print("\n✅ Los documentos están listos para búsqueda.")
                print(f"   Colección utilizada: {collection_name}")
                print(f"   - Partición documentos: documents")
                print(f"   - Partición resúmenes: summaries")
                print("   Ejecuta 'python run_retrieval.py' para probar las búsquedas. (devolverá solo los fragmentos de texto similares a la consulta)")
                print("   Ejecuta 'python run_chatbot.py' para probar el chatbot. (devolverá una respuesta basada en los fragmentos extraidos de la búsqueda)")
            
    except Exception as e:
        logger.error(f"Error en el pipeline: {str(e)}", exc_info=True)
        print(f"\n❌ Error crítico: {str(e)}")
        print("\nAsegúrate de que:")
        print("  1. Milvus está corriendo (docker-compose up -d)")
        print("  2. Las variables de entorno están configuradas (.env)")
        print("  3. Tienes una API key válida de OpenAI")


def main():
    """Función principal"""
    
    print("\n📤 Iniciando subida de documentos para Ejemplo 1: El Principito...\n")
    
    # Verificar que Milvus está disponible
    print("ℹ️  Asegúrate de que Milvus está corriendo:")
    print("   docker-compose up -d\n")
    
    upload_documents()


if __name__ == "__main__":
    main()