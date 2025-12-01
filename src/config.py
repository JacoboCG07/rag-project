"""
Configuration for RAG Project
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Configuración de Milvus
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "default")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "documents")

# Configuración de OpenAI (para embeddings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuración de embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
EMBEDDING_DIM = 1536  # Dimensión para text-embedding-ada-002

# Configuración de chunking
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

