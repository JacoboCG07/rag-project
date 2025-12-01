"""
Base classes for Milvus indices
Implements Strategy pattern for index providers
"""

from abc import ABC, abstractmethod
from typing import Dict


class IndexProvider(ABC):
    """
    Base interface for building Milvus indexes.
    Implements Strategy pattern to allow different index types.
    """

    @abstractmethod
    def build_params(self) -> Dict:
        """
        Builds and returns index parameters for Milvus.

        Returns:
            Dict: Index parameters ready to use in index creation.
        """
        pass

    @abstractmethod
    def get_field_name(self) -> str:
        """
        Returns the field name where the index will be created.

        Returns:
            str: Field name for the index.
        """
        pass
