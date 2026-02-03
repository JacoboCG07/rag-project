"""
HNSW index strategy for Milvus
Hierarchical Navigable Small World
"""

from typing import Dict
from ..base import IndexProvider


class HNSWIndexProvider(IndexProvider):
    """
    Strategy for HNSW index (Hierarchical Navigable Small World).
    Provides faster search with higher memory usage.
    """

    def __init__(self, M: int = 16, ef_construction: int = 200):
        """
        Initializes the HNSW index provider.

        Args:
            M: Number of bi-directional links for each node (default 16).
            ef_construction: Size of dynamic candidate list (default 200).
        """
        self.M = M
        self.ef_construction = ef_construction

    def build_params(self) -> Dict:
        """
        Builds index parameters for HNSW index.

        Returns:
            Dict: Index parameters.
        """
        return {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {
                "M": self.M,
                "ef_construction": self.ef_construction
            }
        }

    def get_field_name(self) -> str:
        """
        Returns the field name for text embeddings.

        Returns:
            str: Field name.
        """
        return "text_embedding"

