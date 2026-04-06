"""
IVF_SQ8 index strategy for Milvus
Inverted File with Scalar Quantization
"""

from typing import Dict
from ..base import IndexProvider


class IVF_SQ8IndexProvider(IndexProvider):
    """
    Strategy for IVF_SQ8 index (Inverted File with Scalar Quantization).
    Provides good balance between memory usage and search speed.
    """

    def __init__(self, nlist: int = 128):
        """
        Initializes the IVF_SQ8 index provider.

        Args:
            nlist: Number of clusters (default 128).
        """
        self.nlist = nlist

    def build_params(self) -> Dict:
        """
        Builds index parameters for IVF_SQ8 index.

        Returns:
            Dict: Index parameters.
        """
        return {
            "index_type": "IVF_SQ8",
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

