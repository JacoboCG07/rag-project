"""
Metadata Filter Builder
Construye expresiones de filtro de Milvus desde metadata extraída
"""
from typing import Dict, List, Any, Optional
from src.utils import get_logger


class MetadataFilterBuilder:
    """
    Construye expresiones de filtro de Milvus desde metadata extraída por el LLM.
    
    Convierte diccionarios de metadata a sintaxis de filtros de Milvus.
    """
    
    def __init__(self):
        """Inicializa el constructor de filtros."""
        self.logger = get_logger(__name__)
        self.logger.info("MetadataFilterBuilder initialized")
    
    def build_filter_for_document(
        self,
        file_id: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Construye una expresión de filtro para un documento específico.
        
        Args:
            file_id: ID del documento.
            metadata: Diccionario con metadata extraída.
                {
                    "pages": [1, 2, 3],
                    "chapters": ["cap1"],
                    "search_image": False,
                    "num_image": None,
                    "type_file": "PDF"
                }
        
        Returns:
            str: Expresión de filtro de Milvus.
            
            Ejemplo:
            'file_id == "doc_001" and pages in ["1", "2", "3"] and type_file == "PDF"'
        """
        self.logger.debug(f"Building filter for document: {file_id}")
        
        filter_parts = []
        
        # Siempre incluir file_id
        filter_parts.append(f'file_id == "{file_id}"')
        
        # Pages
        if metadata.get("pages"):
            pages_filter = self._build_pages_filter(metadata["pages"])
            if pages_filter:
                filter_parts.append(pages_filter)
        
        # Chapters
        if metadata.get("chapters"):
            chapters_filter = self._build_chapters_filter(metadata["chapters"])
            if chapters_filter:
                filter_parts.append(chapters_filter)
        
        # Search image
        if metadata.get("search_image") is not None:
            search_image = metadata["search_image"]
            # TODO: Decidir cómo filtrar por imágenes en Milvus
            pass
        
        # Num image
        if metadata.get("num_image"):
            num_image_filter = self._build_num_image_filter(metadata["num_image"])
            if num_image_filter:
                filter_parts.append(num_image_filter)
        
        # Type file
        if metadata.get("type_file"):
            filter_parts.append(f'type_file == "{metadata["type_file"]}"')
        
        # Combinar con AND
        filter_expr = " and ".join(filter_parts)
        
        self.logger.debug(f"Filter built: {filter_expr}")
        return filter_expr
    
    def build_combined_filter(
        self,
        metadata_dict: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Construye un filtro combinado para múltiples documentos.
        
        Combina los filtros de cada documento con OR.
        
        Args:
            metadata_dict: Diccionario de metadata por file_id.
                {
                    "doc_001": {...},
                    "doc_002": {...}
                }
        
        Returns:
            str: Expresión de filtro combinada.
            
            Ejemplo:
            '(file_id == "doc_001" and pages in ["1","2"]) or 
             (file_id == "doc_002" and chapters in ["cap1"])'
        """
        self.logger.info(
            "Building combined filter",
            extra={"num_documents": len(metadata_dict)}
        )
        
        if not metadata_dict:
            return ""
        
        document_filters = []
        
        for file_id, metadata in metadata_dict.items():
            doc_filter = self.build_filter_for_document(file_id, metadata)
            if doc_filter:
                # Envolver cada filtro de documento en paréntesis
                document_filters.append(f"({doc_filter})")
        
        if not document_filters:
            return ""
        
        # Combinar con OR
        combined_filter = " or ".join(document_filters)
        
        self.logger.info(f"Combined filter created with {len(document_filters)} documents")
        return combined_filter
    
    def build_filter_only_file_ids(
        self,
        file_ids: List[str]
    ) -> str:
        """
        Construye un filtro simple solo con file_ids (sin metadata adicional).
        
        Args:
            file_ids: Lista de IDs de documentos.
        
        Returns:
            str: Expresión de filtro.
            
            Ejemplo:
            'file_id in ["doc_001", "doc_002", "doc_003"]'
        """
        if not file_ids:
            return ""
        
        if len(file_ids) == 1:
            return f'file_id == "{file_ids[0]}"'
        
        file_ids_str = ", ".join([f'"{fid}"' for fid in file_ids])
        return f'file_id in [{file_ids_str}]'
    
    def _build_pages_filter(self, pages: List[int]) -> str:
        """
        Construye filtro para páginas.
        
        Args:
            pages: Lista de números de página.
        
        Returns:
            str: Filtro de páginas.
        """
        # TODO: Implementar lógica de filtrado de páginas
        # Depende de cómo se almacenan las páginas en Milvus
        # Opciones:
        # 1. Campo 'page_number' por chunk
        # 2. Campo 'pages' con rango (ej: "1-5")
        # 3. Otro formato
        
        if not pages:
            return ""
        
        # Ejemplo: 'page_number in [1, 2, 3]'
        pages_str = ", ".join([f'"{p}"' for p in pages])
        return f'pages in [{pages_str}]'
    
    def _build_chapters_filter(self, chapters: List[str]) -> str:
        """
        Construye filtro para capítulos.
        
        Args:
            chapters: Lista de nombres de capítulos.
        
        Returns:
            str: Filtro de capítulos.
        """
        # TODO: Implementar lógica de filtrado de capítulos
        if not chapters:
            return ""
        
        # Ejemplo: 'chapter in ["cap1", "cap2"]'
        chapters_str = ", ".join([f'"{c}"' for c in chapters])
        return f'chapters in [{chapters_str}]'
    
    def _build_num_image_filter(self, num_images: List[int]) -> str:
        """
        Construye filtro para números de imagen.
        
        Args:
            num_images: Lista de números de imagen.
        
        Returns:
            str: Filtro de imágenes.
        """
        # TODO: Implementar lógica de filtrado por número de imagen
        if not num_images:
            return ""
        
        # Ejemplo: 'image_number in [1, 2, 3]'
        images_str = ", ".join([f'"{n}"' for n in num_images])
        return f'num_image in [{images_str}]'
    
    def validate_filter(self, filter_expr: str) -> bool:
        """
        Valida que la expresión de filtro sea sintácticamente correcta.
        
        Args:
            filter_expr: Expresión de filtro a validar.
        
        Returns:
            bool: True si es válida.
        """
        # TODO: Implementar validación básica
        # - No esté vacía
        # - Paréntesis balanceados
        # - Sintaxis básica de Milvus
        
        if not filter_expr:
            return False
        
        # Validación básica de paréntesis
        open_count = filter_expr.count("(")
        close_count = filter_expr.count(")")
        
        return open_count == close_count

