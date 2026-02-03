"""
FLAT index strategy for Milvus
Brute force exact search
"""

from typing import Dict
from ..base import IndexProvider


class FLATIndexProvider(IndexProvider):
    """
    Strategy for FLAT index (brute force).
    Provides exact search results but slower for large datasets.
    """

    def build_params(self) -> Dict:
        """
        Builds index parameters for FLAT index.

        Returns:
            Dict: Index parameters.
        """
        return {
            "index_type": "FLAT",
            "metric_type": "COSINE"
        }

    def get_field_name(self) -> str:
        """
        Returns the field name for text embeddings.

        Returns:
            str: Field name.
        """
        return "text_embedding"

