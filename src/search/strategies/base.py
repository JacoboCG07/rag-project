"""
Base Strategy for Search Operations
Defines the interface for all search strategies
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.utils import get_logger


class SearchStrategy(ABC):
    """
    Abstract base class for search strategies.
    Each concrete strategy implements a different type of search.
    """
    
    def __init__(self, config):
        """
        Initialize the strategy with configuration.
        
        Args:
            config: SearchPipelineConfig with configuration parameters.
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__}")
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        user_query: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs search according to the strategy.
        
        Args:
            query_embedding: Embedding vector of the query.
            user_query: User query text (optional, required by some strategies).
            partition_names: List of partition names to search in (None = all partitions).
            filter_expr: Optional filter expression (e.g., 'file_id == "123"').
            
        Returns:
            List[Dict[str, Any]]: List of documents found with their scores and metadata.
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Closes connections and releases resources.
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.logger.error(
                f"Exception in context manager: {exc_type.__name__}",
                extra={
                    "exception_type": exc_type.__name__,
                    "exception_value": str(exc_val) if exc_val else None
                },
                exc_info=True
            )
        self.close()

