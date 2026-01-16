"""
Metadata Extractor
Extrae metadata de documentos usando LLM basándose en el query del usuario
"""
from typing import Dict, List, Any, Optional
from llms.text import BaseTextModel
from src.utils import get_logger
import json
import os


class MetadataExtractor:
    """
    Extrae metadata relevante de documentos usando un LLM.
    
    Analiza el query del usuario y el markdown de documentos para identificar
    metadata específica como páginas, capítulos, imágenes, etc.
    """
    
    def __init__(
        self,
        text_model: BaseTextModel,
        max_tokens: int = 500,
        temperature: float = 0.2
    ):
        """
        Inicializa el extractor de metadata.
        
        Args:
            text_model: Modelo LLM para extraer metadata.
            max_tokens: Tokens máximos para la respuesta del LLM.
            temperature: Temperatura del LLM (0.0-1.0).
        """
        self.text_model = text_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger = get_logger(__name__)
        
        # Cargar prompt
        self.prompt_template = self._load_prompt()
        
        self.logger.info(
            "MetadataExtractor initialized",
            extra={
                "model": text_model.__class__.__name__,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
    
    def _load_prompt(self) -> str:
        """
        Carga el prompt desde el archivo prompt.md.
        
        Returns:
            str: Template del prompt.
        """
        prompt_path = os.path.join(os.path.dirname(__file__), "prompt.md")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            self.logger.warning(f"Prompt file not found: {prompt_path}")
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """
        Devuelve un prompt por defecto si no se encuentra el archivo.
        
        Returns:
            str: Prompt por defecto.
        """
        return """Analiza la consulta del usuario y extrae metadata relevante.

Documentos:
{markdown_documents}

Consulta: {user_query}

Extrae para cada documento:
- pages: Lista de números de página (o null)
- chapters: Lista de capítulos (o null)
- search_image: true/false si busca imágenes
- num_image: Lista de números de imagen (o null)
- type_file: Tipo de archivo

Responde SOLO con JSON válido sin explicaciones."""
    
    def extract(
        self,
        user_query: str,
        markdown_documents: str,
        documents_info: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extrae metadata de los documentos basándose en el query del usuario.
        
        Args:
            user_query: Consulta del usuario.
            markdown_documents: Markdown con información de documentos.
            documents_info: Lista de diccionarios con info de documentos.
        
        Returns:
            Dict[str, Dict]: Metadata extraída por file_id.
            
            Ejemplo:
            {
                "doc_001": {
                    "pages": [1, 2, 3],
                    "chapters": null,
                    "search_image": False,
                    "num_image": null,
                    "type_file": "PDF"
                }
            }
        """
        self.logger.info(
            "Extracting metadata from documents",
            extra={
                "user_query": user_query,
                "num_documents": len(documents_info)
            }
        )
        
        try:
            # Construir prompt
            prompt = self._build_prompt(user_query, markdown_documents, documents_info)
            
            # Llamar al LLM
            self.logger.debug("Calling LLM for metadata extraction")
            response = self.text_model.generate(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Parsear respuesta
            metadata_dict = self._parse_response(response, documents_info)
            
            self.logger.info(
                "Metadata extraction completed",
                extra={
                    "documents_with_metadata": len(metadata_dict)
                }
            )
            
            return metadata_dict
            
        except Exception as e:
            self.logger.error(
                f"Error extracting metadata: {str(e)}",
                extra={"error_type": type(e).__name__},
                exc_info=True
            )
            raise
    
    def _build_prompt(
        self,
        user_query: str,
        markdown_documents: str,
        documents_info: List[Dict[str, Any]]
    ) -> str:
        """
        Construye el prompt para el LLM.
        
        Args:
            user_query: Consulta del usuario.
            markdown_documents: Markdown de documentos.
            documents_info: Información de documentos.
        
        Returns:
            str: Prompt completo.
        """
        # TODO: Implementar construcción del prompt
        # Reemplazar placeholders en self.prompt_template
        prompt = self.prompt_template.format(
            user_query=user_query,
            markdown_documents=markdown_documents
        )
        return prompt
    
    def _parse_response(
        self,
        response: str,
        documents_info: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Parsea la respuesta del LLM y valida los datos.
        
        Args:
            response: Respuesta del LLM (JSON string).
            documents_info: Info de documentos para validación.
        
        Returns:
            Dict[str, Dict]: Metadata parseada y validada.
        """
        try:
            # TODO: Implementar parsing
            # 1. Extraer JSON de la respuesta
            # 2. Validar estructura
            # 3. Validar valores contra documents_info
            # 4. Devolver dict limpio
            
            metadata_dict = json.loads(response)
            return metadata_dict
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            return {}
        except Exception as e:
            self.logger.error(f"Error parsing response: {str(e)}")
            return {}
    
    def _validate_metadata(
        self,
        file_id: str,
        metadata: Dict[str, Any],
        document_info: Dict[str, Any]
    ) -> bool:
        """
        Valida que la metadata extraída sea consistente con el documento.
        
        Args:
            file_id: ID del documento.
            metadata: Metadata extraída.
            document_info: Información real del documento.
        
        Returns:
            bool: True si es válida.
        """
        # TODO: Implementar validaciones
        # - Páginas no excedan total_pages
        # - Capítulos estén en rango válido
        # - Imágenes no excedan total_num_image
        # - etc.
        return True

