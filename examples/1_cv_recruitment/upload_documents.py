"""
Script de Subida de Documentos - Ejemplo 1: CVs y Propuesta de Trabajo
========================================================================

Este script indexa los documentos del ejemplo en Milvus usando el RAG Pipeline.

Documentos a subir:
- job_proposal.pdf
- cv_candidate_1.pdf
- cv_candidate_2.pdf
- cv_candidate_3.pdf
- cv_candidate_4.pdf
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

from src.rag.rag_pipeline import RAGPipeline
from src.rag.config import RAGPipelineConfig
from src.utils import get_logger

logger = get_logger(__name__)


def upload_documents():
    """Sube e indexa todos los documentos del ejemplo en Milvus"""
    
    print("=" * 80)
    print("SUBIDA DE DOCUMENTOS - EJEMPLO 1: CVs Y PROPUESTA DE TRABAJO")
    print("=" * 80)
    
    # Ruta a la carpeta de datos
    data_dir = Path(__file__).parent / "data"
    
    # Lista de documentos a procesar
    expected_files = [
        "job_proposal.pdf",
        "cv_candidate_1.pdf",
        "cv_candidate_2.pdf",
        "cv_candidate_3.pdf",
        "cv_candidate_4.pdf"
    ]
    
    # Obtener rutas de los archivos
    existing_files = [data_dir / filename for filename in expected_files]
    
    print(f"\n🚀 Procesando {len(existing_files)} documento(s)...\n")
    
    # Nombre de la colección para este ejemplo
    # La colección tendrá dos particiones: 'documents' y 'summaries'
    collection_name = "cv_recruitment"
    
    try:
        # Configurar el RAG Pipeline
        # Nota: Ajusta la configuración según tus necesidades
        config = RAGPipelineConfig(collection_name=collection_name)
        
        with RAGPipeline(config=config) as pipeline:
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
                        extract_process_images=False
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
                print("   Ejecuta 'python run_example.py' para probar las búsquedas.")
            
    except Exception as e:
        logger.error(f"Error en el pipeline: {str(e)}", exc_info=True)
        print(f"\n❌ Error crítico: {str(e)}")
        print("\nAsegúrate de que:")
        print("  1. Milvus está corriendo (docker-compose up -d)")
        print("  2. Las variables de entorno están configuradas (.env)")
        print("  3. Tienes una API key válida de OpenAI")


def main():
    """Función principal"""
    
    print("\n📤 Iniciando subida de documentos para Ejemplo 1: CVs...\n")
    
    # Verificar que Milvus está disponible
    print("ℹ️  Asegúrate de que Milvus está corriendo:")
    print("   docker-compose up -d\n")
    
    upload_documents()


if __name__ == "__main__":
    main()