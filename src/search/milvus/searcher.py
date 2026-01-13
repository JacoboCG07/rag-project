"""
Módulo para buscar documentos en Milvus
"""
from typing import List, Dict, Optional
import sys
import os

# Agregar la ruta de libraries-main si existe
libraries_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "libraries-main")
if os.path.exists(libraries_path):
    sys.path.insert(0, libraries_path)

try:
    from firemilvus import controlMilvus
except ImportError:
    controlMilvus = None

from pymilvus import Collection, connections, db, utility


class MilvusSearcher:
    """Clase para buscar documentos en Milvus"""
    
    def __init__(self, db_name: str, collection_name: str, alias: str = "default"):
        """
        Inicializa el buscador de Milvus
        
        Args:
            db_name: Nombre de la base de datos
            collection_name: Nombre de la colección
            alias: Alias para la conexión
        """
        self.db_name = db_name
        self.collection_name = collection_name
        self.alias = alias
        self.collection = None
    
    def connect(self):
        """Conecta a Milvus y carga la colección"""
        if controlMilvus:
            self.collection = controlMilvus.load_db_and_collection(
                dbname=self.db_name,
                collection_name=self.collection_name,
                alias=self.alias
            )
        else:
            # Implementación alternativa usando pymilvus directamente
            from dotenv import load_dotenv
            load_dotenv()
            
            host = os.getenv("MILVUS_HOST", "localhost")
            port = os.getenv("MILVUS_PORT", "19530")
            
            connections.connect(host=host, port=port, alias=self.alias)
            
            # Listar bases de datos y crear si no existe
            dbs = db.list_database(using=self.alias)
            if self.db_name not in dbs:
                db.create_database(self.db_name, using=self.alias)
            db.using_database(self.db_name, using=self.alias)
            
            self.collection = Collection(self.collection_name, using=self.alias)
    
    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        partition_names: Optional[List[str]] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict]:
        """
        Busca documentos similares en Milvus
        
        Args:
            query_embedding: Embedding de la query
            limit: Número máximo de resultados
            partition_names: Lista de particiones donde buscar (None = todas)
            filter_expr: Expresión de filtro opcional (ej: 'file_id == "123"')
            
        Returns:
            Lista de documentos encontrados con sus scores
        """
        try:
            # Preparar parámetros de búsqueda
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Realizar búsqueda
            results = self.collection.search(
                data=[query_embedding],
                anns_field="text_embedding",
                param=search_params,
                limit=limit,
                partition_names=partition_names,
                expr=filter_expr,
                output_fields=["text", "file_id", "file_name", "source_id", "pages", "chapters", "type_file"]
            )
            
            # Procesar resultados
            documents = []
            if results:
                for hit in results[0]:
                    doc = {
                        "id": hit.id,
                        "score": hit.score,
                        "text": hit.entity.get("text", ""),
                        "file_id": hit.entity.get("file_id", ""),
                        "file_name": hit.entity.get("file_name", ""),
                        "source_id": hit.entity.get("source_id", ""),
                        "pages": hit.entity.get("pages", ""),
                        "chapters": hit.entity.get("chapters", ""),
                        "type_file": hit.entity.get("type_file", "")
                    }
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            raise Exception(f"Error buscando en Milvus: {str(e)}")
    
    def search_by_partition(
        self,
        query_embedding: List[float],
        partition_name: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Busca documentos en una partición específica
        
        Args:
            query_embedding: Embedding de la query
            partition_name: Nombre de la partición
            limit: Número máximo de resultados
            
        Returns:
            Lista de documentos encontrados
        """
        return self.search(
            query_embedding=query_embedding,
            limit=limit,
            partition_names=[partition_name]
        )
    
    def get_partitions(self) -> List[str]:
        """
        Obtiene la lista de particiones disponibles en la colección
        
        Returns:
            Lista de nombres de particiones
        """
        try:
            partitions = self.collection.partitions
            # Filtrar la partición default
            partition_names = [p.name for p in partitions if p.name != "_default"]
            return partition_names
        except Exception as e:
            raise Exception(f"Error obteniendo particiones: {str(e)}")
    
    def disconnect(self):
        """Cierra la conexión con Milvus"""
        if controlMilvus:
            controlMilvus.finish_conection_and_release_collection(
                alias=self.alias,
                collection=self.collection
            )
        else:
            self.collection.release()
            connections.disconnect(alias=self.alias)

