"""
Módulo para buscar documentos en Milvus
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
from pymilvus import Collection, connections, db

load_dotenv()

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
        """Conecta a Milvus y carga la colección. Idempotente: si ya está conectado y la colección cargada, no hace nada."""
        if self.collection is not None:
            return
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")

        connections.connect(host=host, port=port, alias=self.alias)

        dbs = db.list_database(using=self.alias)
        if self.db_name not in dbs:
            db.create_database(self.db_name, using=self.alias)
        db.using_database(self.db_name, using=self.alias)

        self.collection = Collection(self.collection_name, using=self.alias)
        self.collection.load()  # Cargar en memoria para permitir búsquedas

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
                output_fields=["text", "file_id", "file_name", "file_type", "pages", "chapters"]
            )
            
            # Procesar resultados (pymilvus 2.4+: Hit.entity.get(key) no admite default)
            def _field(entity, name: str) -> str:
                try:
                    val = entity.get(name) if hasattr(entity, "get") else getattr(entity, name, None)
                    return val if val is not None else ""
                except (TypeError, AttributeError):
                    return ""

            documents = []
            if results:
                for hit in results[0]:
                    entity = getattr(hit, "entity", hit)
                    file_id = _field(entity, "file_id")
                    doc = {
                        "id": hit.id,
                        "score": hit.score,
                        "text": _field(entity, "text"),
                        "file_id": file_id,
                        "file_name": _field(entity, "file_name"),
                        "source_id": file_id,
                        "pages": _field(entity, "pages"),
                        "chapters": _field(entity, "chapters"),
                        "file_type": _field(entity, "file_type"),
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
        """
        Libera la colección de memoria. No cierra la conexión global (alias), porque
        en estrategias con selección (DocumentSelector) el SummaryRetriever comparte
        la misma conexión y debe seguir usándola en consultas siguientes.
        La conexión se cierra al cerrar el pipeline/estrategia.
        """
        if self.collection is not None:
            self.collection.release()
            self.collection = None
        # No llamar a connections.disconnect(): la conexión es compartida con SummaryRetriever

