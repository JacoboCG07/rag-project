"""
Simple Search Strategy
Performs direct vector search in Milvus without document selection
"""
from typing import List, Dict, Any, Optional
from .base import SearchStrategy
from src.search.milvus import MilvusSearcher


class SimpleSearchStrategy(SearchStrategy):
    """
    Simple search strategy: Direct vector search in Milvus.
    No document selection, just pure similarity search.
    """
    
    def __init__(self, config):
        """
        Initialize the simple search strategy.
        
        Args:
            config: SearchPipelineConfig with configuration parameters.
        """
        super().__init__(config)
        
        # Initialize MilvusSearcher
        self.searcher = MilvusSearcher(
            db_name=config.milvus.dbname,
            collection_name=config.collection_name_documents,
            alias=config.milvus.alias
        )
        
        self.logger.info(
            "SimpleSearchStrategy initialized",
            extra={
                "collection": config.collection_name_documents,
                "search_limit": config.search_limit
            }
        )
    
    def search(
        self,
        query_embedding: List[float],
        user_query: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs simple vector search in Milvus.
        
        Args:
            query_embedding: Embedding vector of the query.
            user_query: Not used in simple search (optional).
            partition_names: List of partition names to search in.
            filter_expr: Optional filter expression.
            
        Returns:
            List[Dict[str, Any]]: Search results.
        """
        self.logger.info(
            "Executing simple search",
            extra={
                "has_partition_names": partition_names is not None,
                "has_filter_expr": filter_expr is not None
            }
        )
        
        try:
            # Connect to Milvus
            self.searcher.connect()
            
            # Perform search
            results = self.searcher.search(
                query_embedding=query_embedding,
                limit=self.config.search_limit,
                partition_names=partition_names,
                filter_expr=filter_expr
            )
            
            self.logger.info(
                "Simple search completed",
                extra={
                    "results_count": len(results),
                    "search_limit": self.config.search_limit
                }
            )
            
            return results
            
        except Exception as e:
            self.logger.error(
                f"Error in simple search: {str(e)}",
                extra={"error_type": type(e).__name__},
                exc_info=True
            )
            raise
        finally:
            # Disconnect from Milvus
            try:
                self.searcher.disconnect()
            except Exception as e:
                self.logger.warning(f"Error disconnecting from Milvus: {str(e)}")
    
    def close(self) -> None:
        """
        Closes connections with Milvus.
        """
        self.logger.info("Closing SimpleSearchStrategy connections")
        try:
            # Note: searcher.disconnect() is called after each search
            self.logger.info("SimpleSearchStrategy closed successfully")
        except Exception as e:
            self.logger.error(
                f"Error closing SimpleSearchStrategy: {str(e)}",
                extra={"error_type": type(e).__name__},
                exc_info=True
            )

