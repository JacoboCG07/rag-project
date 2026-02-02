"""
Search Pipeline
Executes search operations using different strategies
"""
from typing import List, Dict, Any, Optional
from src.search.config import SearchPipelineConfig, SearchType
from src.search.strategies import (
    SearchStrategy,
    SimpleSearchStrategy,
    DocumentSelectorSearchStrategy,
    DocumentSelectorMetadataSearchStrategy
)
from src.utils import get_logger


class SearchPipeline:
    """
    High-level search pipeline that uses different strategies:
    - Simple search: Direct vector search in Milvus
    - Search with selection: First selects documents, then searches within them
    - Search with selection and metadata: Document selection + metadata filters
    
    Uses the Strategy pattern to delegate search operations.
    """
    
    def __init__(
        self,
        *,
        config: SearchPipelineConfig
    ):
        """
        Initializes the Search Pipeline with the appropriate strategy.
        
        Args:
            config: SearchPipelineConfig with all configuration parameters.
        """
        self.logger = get_logger(__name__)
        self.config = config
        
        self.logger.info(
            "Initializing Search Pipeline",
            extra={
                "search_type": config.search_type.value,
                "milvus_db": config.milvus.dbname,
                "collection_name": config.collection_name,
                "partition_documents": config.PARTITION_DOCUMENTS,
                "partition_summaries": config.PARTITION_SUMMARIES,
                "search_limit": config.search_limit
            }
        )
        
        # Create the appropriate strategy based on search type
        self.strategy = self._create_strategy(config)
        
        self.logger.info(
            f"Search Pipeline initialized with {self.strategy.__class__.__name__}"
        )
    
    def _create_strategy(self, config: SearchPipelineConfig) -> SearchStrategy:
        """
        Factory method to create the appropriate search strategy.
        
        Args:
            config: SearchPipelineConfig with configuration.
            
        Returns:
            SearchStrategy: Concrete strategy instance.
            
        Raises:
            ValueError: If search_type is unknown.
        """
        if config.search_type == SearchType.SIMPLE:
            return SimpleSearchStrategy(config)
        
        elif config.search_type == SearchType.WITH_SELECTION:
            return DocumentSelectorSearchStrategy(config)
        
        elif config.search_type == SearchType.WITH_SELECTION_AND_METADATA:
            return DocumentSelectorMetadataSearchStrategy(config)
        
        else:
            raise ValueError(f"Unknown search_type: {config.search_type}")
    
    def search(
        self,
        query_embedding: List[float],
        user_query: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs search using the configured strategy.
        
        Args:
            query_embedding: Embedding vector of the query.
            user_query: User query text (required for strategies with document selection).
            partition_names: List of partition names to search in (None = all partitions).
            filter_expr: Optional filter expression (e.g., 'file_id == "123"').
            
        Returns:
            List[Dict[str, Any]]: List of documents found with their scores and metadata.
            
        Raises:
            ValueError: If required parameters are missing for the selected strategy.
        """
        self.logger.info(
            "Starting search",
            extra={
                "search_type": self.config.search_type.value,
                "strategy": self.strategy.__class__.__name__,
                "has_partition_names": partition_names is not None,
                "has_filter_expr": filter_expr is not None,
                "has_user_query": user_query is not None
            }
        )
        
        # Delegate to the strategy
        return self.strategy.search(
            query_embedding=query_embedding,
            user_query=user_query,
            partition_names=partition_names,
            filter_expr=filter_expr
        )
    
    def close(self) -> None:
        """Closes connections with the strategy."""
        self.logger.info("Closing Search Pipeline connections")
        try:
            if self.strategy is not None:
                self.strategy.close()
            self.logger.info("Search Pipeline connections closed successfully")
        except Exception as e:
            self.logger.error(
                f"Error closing Search Pipeline connections: {str(e)}",
                extra={"error_type": type(e).__name__},
                exc_info=True
            )
    
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
