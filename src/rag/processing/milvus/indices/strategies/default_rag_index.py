"""
Default RAG index strategy for Milvus
IVF_FLAT with COSINE metric
"""

from typing import Dict
from ..base import IndexProvider


class DefaultRAGIndexProvider(IndexProvider):
    """
    Strategy for default RAG index (IVF_FLAT with COSINE metric).
    Optimized for Retrieval Augmented Generation with good balance
    between search speed and accuracy.
    """

    def __init__(self, nlist: int = 128):
        """
        Initializes the default RAG index provider.

        Args:
            nlist: Number of clusters for IVF_FLAT index (default 128).
        """
        self.nlist = nlist

    def build_params(self) -> Dict:
        """
        Builds index parameters for default RAG index.

        Returns:
            Dict: Index parameters.
        """
        return {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": self.nlist}
        }

    def get_field_name(self) -> str:
        """
        Returns the field name for text embeddings.

        Returns:
            str: Field name.
        """
        return "text_embedding"

