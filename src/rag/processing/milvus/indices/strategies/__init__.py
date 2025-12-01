"""
Index strategies for Milvus
"""

from .default_rag_index import DefaultRAGIndexProvider
from .hnsw_index import HNSWIndexProvider
from .ivf_sq8_index import IVF_SQ8IndexProvider
from .flat_index import FLATIndexProvider

__all__ = [
    'DefaultRAGIndexProvider',
    'HNSWIndexProvider',
    'IVF_SQ8IndexProvider',
    'FLATIndexProvider'
]

