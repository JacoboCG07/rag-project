"""
Script de Subida de Documentos - Ejemplo 1: CVs y Propuesta de Trabajo
========================================================================

Este script indexa los documentos del ejemplo en Milvus usando el RAG Pipeline.

Documentos a subir:
- job_proposal.pdf
- cv_candidate_1.pdf
- cv_candidate_2.pdf
- cv_candidate_3.pdf
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
    """Sube e indexa todos los documentos del ejemplo en Milvus"""
    
    print("=" * 80)
    print("SUBIDA DE DOCUMENTOS - EJEMPLO 1: CVs Y PROPUESTA DE TRABAJO")
    print("=" * 80)
    
    # Ruta a la carpeta de datos
    data_dir = Path(__file__).parent / "data"
    
    # Lista de documentos esperados
    expected_files = [
        "job_proposal.pdf",
        "cv_candidate_1.pdf",
        "cv_candidate_2.pdf",
        "cv_candidate_3.pdf"
    ]
    
    # Verificar que existen los archivos
    print("\n📋 Verificando archivos...")
    missing_files = []
    existing_files = []
    
    for filename in expected_files:
        file_path = data_dir / filename
        if file_path.exists():
            existing_files.append(file_path)
            print(f"  ✓ {filename}")
        else:
            missing_files.append(filename)
            print(f"  ✗ {filename} - NO ENCONTRADO")
    
    if missing_files:
        print(f"\n⚠️  ADVERTENCIA: Faltan {len(missing_files)} archivo(s):")
        for f in missing_files:
            print(f"     - {f}")
        print("\nPor favor, añade los archivos faltantes a la carpeta 'data/'")
        
        response = input("\n¿Deseas continuar con los archivos disponibles? (s/n): ")
        if response.lower() != 's':
            print("Subida cancelada.")
            return
    
    if not existing_files:
        print("\n❌ No hay archivos para procesar. Añade los PDFs a la carpeta 'data/'")
        return
    
    print(f"\n🚀 Procesando {len(existing_files)} documento(s)...\n")
    
    try:
        # Configurar el RAG Pipeline
        # Nota: Ajusta la configuración según tus necesidades
        config = RAGPipelineConfig()
        
        with RAGPipeline(config=config) as pipeline:
            successful = 0
            failed = 0
            
            for i, file_path in enumerate(existing_files, 1):
                print(f"\n{'─' * 80}")
                print(f"[{i}/{len(existing_files)}] Procesando: {file_path.name}")
                print('─' * 80)
                
                try:
                    # Procesar e indexar el documento
                    result = pipeline.process_document(str(file_path))
                    
                    print(f"✓ {file_path.name} procesado correctamente")
                    if result:
                        print(f"  - Chunks generados: {result.get('chunks_count', 'N/A')}")
                        print(f"  - File ID: {result.get('file_id', 'N/A')}")
                    
                    successful += 1
                    
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
    
    response = input("¿Milvus está corriendo? (s/n): ")
    if response.lower() != 's':
        print("\nPor favor, inicia Milvus primero:")
        print("  cd ../../  # Ir a la raíz del proyecto")
        print("  docker-compose up -d")
        return
    
    upload_documents()


if __name__ == "__main__":
    main()

