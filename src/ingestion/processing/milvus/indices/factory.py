"""
Factory for Milvus indices
Implements Factory pattern for index creation
"""

from typing import Dict, Type, Optional, Tuple

from .base import IndexProvider
from .exceptions import IndexNotFoundError
from .strategies.default_rag_index import DefaultRAGIndexProvider
from .strategies.hnsw_index import HNSWIndexProvider
from .strategies.ivf_sq8_index import IVF_SQ8IndexProvider
from .strategies.flat_index import FLATIndexProvider


class Indices:
    """
    Factory for Milvus indexes.
    Resolves by name and delegates to the appropriate strategy.
    Implements Factory pattern for index creation.
    """

    _registry: Dict[str, Type[IndexProvider]] = {
        "default": DefaultRAGIndexProvider,
        "ivf_flat": DefaultRAGIndexProvider,
        "hnsw": HNSWIndexProvider,
        "ivf_sq8": IVF_SQ8IndexProvider,
        "flat": FLATIndexProvider,
    }

    def get_index(
        self,
        *,
        name_index: str,
        nlist: Optional[int] = None,
        M: Optional[int] = None,
        ef_construction: Optional[int] = None
    ) -> Tuple[Dict, str]:
        """
        Returns the index parameters and field name for the given index name.

        Args:
            name_index: Index name ('default', 'hnsw', 'ivf_sq8', 'flat').
            nlist: Number of clusters for IVF indexes (optional).
            M: M parameter for HNSW index (optional).
            ef_construction: ef_construction parameter for HNSW index (optional).

        Returns:
            Tuple[Dict, str]: (index parameters, field name).

        Raises:
            IndexNotFoundError: If no provider exists for name_index.
        """
        key = name_index.strip().lower()
        provider_cls = self._registry.get(key)
        
        if not provider_cls:
            raise IndexNotFoundError(
                f"No index found for '{name_index}'. "
                f"Valid options: {list(self._registry.keys())}"
            )
        
        # Initialize provider with appropriate parameters
        if provider_cls == DefaultRAGIndexProvider:
            provider = provider_cls(nlist=nlist or 128)
        elif provider_cls == HNSWIndexProvider:
            provider = provider_cls(
                M=M or 16,
                ef_construction=ef_construction or 200
            )
        elif provider_cls == IVF_SQ8IndexProvider:
            provider = provider_cls(nlist=nlist or 128)
        else:
            provider = provider_cls()
        
        return provider.build_params(), provider.get_field_name()

