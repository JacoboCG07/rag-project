"""
Milvus partition names: the only source of truth is environment variables (.env).
They are not exposed via Pydantic settings or constructor parameters.
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
