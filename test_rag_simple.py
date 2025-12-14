"""
Script simple para probar la inserción de datos en RAG Pipeline
"""
import sys
from pathlib import Path

# Agregar src al path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from rag.config import RAGPipelineConfig
from rag.rag_pipeline import RAGPipeline


def main():
    """Prueba simple de inserción de datos"""
    
    print("=" * 60)
    print("PRUEBA SIMPLE DE RAG PIPELINE")
    print("=" * 60)
    
    # 1. Crear configuración
    print("\n1. Creando configuración...")
    try:
        config = RAGPipelineConfig()
        print(f"   [OK] Base de datos: {config.milvus.dbname}")
        print(f"   [OK] Host: {config.milvus.host}:{config.milvus.port}")
        print(f"   [OK] Collection documentos: {config.collection_name_documents}")
        print(f"   [OK] Chunk size: {config.chunk_size}")
    except Exception as e:
        print(f"   [ERROR] Error creando configuracion: {e}")
        return
    
    # 2. Crear pipeline
    print("\n2. Inicializando pipeline...")
    try:
        pipeline = RAGPipeline(config=config)
        print("   [OK] Pipeline creado")
    except Exception as e:
        print(f"   [ERROR] Error creando pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Seleccionar archivo de prueba
    test_file = project_root / "tests" / "fixtures" / "sample.txt"
    
    if not test_file.exists():
        print(f"\n[ERROR] Error: No se encontro el archivo {test_file}")
        print("   Asegúrate de que existe tests/fixtures/sample.txt")
        pipeline.close()
        return
    
    print(f"\n3. Procesando archivo: {test_file.name}")
    print(f"   Ruta: {test_file}")
    
    # 4. Procesar archivo
    try:
        success, message, info = pipeline.process_single_file(
            file_path=str(test_file),
            extract_process_images=False,
            partition_name="test_partition"
        )
        
        # 5. Mostrar resultados
        print("\n" + "=" * 60)
        print("RESULTADOS")
        print("=" * 60)
        print(f"[OK] Exito: {success}")
        print(f"[OK] Mensaje: {message}")
        print(f"\n[OK] Informacion del archivo:")
        print(f"   - File ID: {info['file_id']}")
        print(f"   - File Name: {info['file_name']}")
        print(f"   - File Path: {info['file_path']}")
        
        if success:
            print("\n[SUCCESS] Insercion completada exitosamente!")
            print(f"   El documento '{info['file_name']}' ha sido procesado")
            print(f"   y los chunks han sido insertados en Milvus.")
        else:
            print(f"\n[ERROR] Error durante la insercion: {message}")
            
    except Exception as e:
        print(f"\n[ERROR] Error inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 6. Cerrar conexiones
        print("\n4. Cerrando conexiones...")
        try:
            pipeline.close()
            print("   [OK] Conexiones cerradas")
        except:
            pass
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Verificar variables de entorno
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("\nVerificando configuración...")
    if not os.getenv("OPENAI_API_KEY"):
        print("[WARNING] Advertencia: OPENAI_API_KEY no esta configurada")
        print("   Los embeddings no funcionarán sin esta clave")
    else:
        print("[OK] OPENAI_API_KEY configurada")
    
    if not os.getenv("MILVUS_HOST"):
        print("[INFO] Usando configuracion por defecto de Milvus (localhost:19530)")
    else:
        print(f"[OK] MILVUS_HOST: {os.getenv('MILVUS_HOST')}")
    
    print()
    
    # Ejecutar prueba
    main()