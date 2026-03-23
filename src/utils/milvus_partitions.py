"""
Nombres de particiones Milvus: única fuente son las variables de entorno (.env).
No se exponen por configuración Pydantic ni por parámetros de constructores.
"""
import os
from dotenv import load_dotenv

load_dotenv()

_ENV_DOCUMENTS = "MILVUS_PARTITION_NAME_DOCUMENTS"
_ENV_SUMMARIES = "MILVUS_PARTITION_NAME_SUMMARIES"


def partition_documents() -> str:
    return os.getenv(_ENV_DOCUMENTS, "documents")


def partition_summaries() -> str:
    return os.getenv(_ENV_SUMMARIES, "summaries")
