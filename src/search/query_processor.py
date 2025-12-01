"""
Módulo para procesar queries y convertirlas en embeddings
"""
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


class QueryProcessor:
    """Clase para procesar queries de búsqueda"""
    
    def __init__(self, model: str = "text-embedding-ada-002"):
        """
        Inicializa el procesador de queries
        
        Args:
            model: Nombre del modelo de embeddings a usar
        """
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY no está configurada en las variables de entorno")
    
    def process_query(self, query: str) -> List[float]:
        """
        Convierte una query en un embedding
        
        Args:
            query: Texto de la consulta
            
        Returns:
            Vector de embedding de la query
        """
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            
            response = client.embeddings.create(
                model=self.model,
                input=query
            )
            
            return response.data[0].embedding
            
        except ImportError:
            raise ImportError("openai no está instalado. Instálalo con: pip install openai")
        except Exception as e:
            raise Exception(f"Error procesando query: {str(e)}")

