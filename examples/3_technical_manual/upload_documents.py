"""
Script de Subida de Documentos - Ejemplo 3: Manual Técnico
============================================================

Este script indexa el manual técnico del ejemplo en Milvus usando el RAG Pipeline.

Documento a subir:
- manual.pdf
"""

import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path para imports
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.ingestion.config import IngestionPipelineConfig
from src.utils import get_logger

logger = get_logger(__name__)


def upload_documents():
    """Sube e indexa el manual técnico en Milvus"""
    
    print("=" * 80)
    print("SUBIDA DE DOCUMENTOS - EJEMPLO 3: MANUAL TÉCNICO")
    print("=" * 80)
    
    # Ruta a la carpeta de datos
    data_dir = Path(__file__).parent / "data"
    manual_file = data_dir / "manual.pdf"
    
    # Verificar que existe el archivo
    print("\n📋 Verificando archivo...")
    
    if not manual_file.exists():
        print(f"  ✗ manual.pdf - NO ENCONTRADO")
        print("\n❌ No se encontró el archivo 'manual.pdf'")
        print("   Por favor, añade el manual a la carpeta 'data/'")
        return
    
    print(f"  ✓ manual.pdf")
    print(f"\n🚀 Procesando manual técnico...\n")
    
    try:
        # Configurar el RAG Pipeline
        # Nota: Ajusta la configuración según tus necesidades
        config = IngestionPipelineConfig()
        
        with IngestionPipeline(config=config) as pipeline:
            print(f"{'─' * 80}")
            print(f"Procesando: {manual_file.name}")
            print('─' * 80)
            
            # Procesar e indexar el documento
            result = pipeline.process_document(str(manual_file))
            
            print(f"\n✓ {manual_file.name} procesado correctamente")
            if result:
                print(f"  - Chunks generados: {result.get('chunks_count', 'N/A')}")
                print(f"  - File ID: {result.get('file_id', 'N/A')}")
                print(f"  - Páginas totales: {result.get('total_pages', 'N/A')}")
            
            print("\n" + "=" * 80)
            print("✅ MANUAL INDEXADO CORRECTAMENTE")
            print("=" * 80)
            print("\nEl manual está listo para búsqueda simple.")
            print("Ejecuta 'python run_example.py' para probar búsquedas directas.")
            
    except Exception as e:
        logger.error(f"Error procesando el manual: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        print("\nAsegúrate de que:")
        print("  1. Milvus está corriendo (docker-compose up -d)")
        print("  2. Las variables de entorno están configuradas (.env)")
        print("  3. Tienes una API key válida de OpenAI")


def main():
    """Función principal"""
    
    print("\n📤 Iniciando subida de documento para Ejemplo 3: Manual Técnico...\n")
    
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

