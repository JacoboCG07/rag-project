"""
Search Pipeline
Executes search operations in Milvus with optional document selection
"""
from typing import List, Dict, Any, Optional
from src.search.config import SearchPipelineConfig, SearchType
from src.search.milvus import MilvusSearcher
from src.search.document_selection import DocumentSelector
from src.utils import get_logger


class SearchPipeline:
    """
    High-level search pipeline that can perform:
    - Normal search: Direct vector search in Milvus
    - Search with selection: First selects relevant documents, then searches within them
    """
    
    def __init__(
        self,
        *,
        config: SearchPipelineConfig
    ):
        """
        Initializes the Search Pipeline.
        
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
                "collection_documents": config.collection_name_documents,
                "collection_summaries": config.collection_name_summaries,
                "search_limit": config.search_limit
            }
        )
        
        # Initialize MilvusSearcher for document search
        self.searcher = MilvusSearcher(
            db_name=config.milvus.dbname,
            collection_name=config.collection_name_documents,
            alias=config.milvus.alias
        )
        
        # Initialize DocumentSelector if needed (for WITH_SELECTION mode)
        self.document_selector: Optional[DocumentSelector] = None
        if config.search_type == SearchType.WITH_SELECTION:
            if config.text_model is None:
                raise ValueError(
                    "text_model is required when search_type='with_selection'. "
                    "Please configure it in SearchPipelineConfig."
                )
            
            self.document_selector = DocumentSelector(
                dbname=config.milvus.dbname,
                collection_name=config.collection_name_summaries,
                text_model=config.text_model,
                uri=config.milvus.uri,
                token=config.milvus.token,
                host=config.milvus.host,
                port=config.milvus.port
            )
            self.logger.info("DocumentSelector initialized for document selection")
        
        self.logger.info("Search Pipeline initialized successfully")
    
    def search(
        self,
        query_embedding: List[float],
        user_query: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs search according to the configured search type.
        
        Args:
            query_embedding: Embedding vector of the query.
            user_query: User query text (required when search_type='with_selection').
            partition_names: List of partition names to search in (None = all partitions).
            filter_expr: Optional filter expression (e.g., 'file_id == "123"').
            
        Returns:
            List[Dict[str, Any]]: List of documents found with their scores and metadata.
            
        Raises:
            ValueError: If user_query is required but not provided.
        """
        self.logger.info(
            "Starting search",
            extra={
                "search_type": self.config.search_type.value,
                "has_partition_names": partition_names is not None,
                "has_filter_expr": filter_expr is not None,
                "has_user_query": user_query is not None
            }
        )
        
        if self.config.search_type == SearchType.NORMAL:
            return self._normal_search(
                query_embedding=query_embedding,
                partition_names=partition_names,
                filter_expr=filter_expr
            )
        
        elif self.config.search_type == SearchType.WITH_SELECTION:
            if user_query is None:
                raise ValueError(
                    "user_query is required when search_type='with_selection'. "
                    "Please provide the user's query text."
                )
            return self._search_with_selection(
                query_embedding=query_embedding,
                user_query=user_query,
                partition_names=partition_names,
                filter_expr=filter_expr
            )
        
        else:
            raise ValueError(f"Unknown search_type: {self.config.search_type}")
    
    def _normal_search(
        self,
        query_embedding: List[float],
        partition_names: Optional[List[str]] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs normal vector search in Milvus.
        
        Args:
            query_embedding: Embedding vector of the query.
            partition_names: List of partition names to search in.
            filter_expr: Optional filter expression.
            
        Returns:
            List[Dict[str, Any]]: Search results.
        """
        self.logger.debug("Executing normal search in Milvus")
        
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
                "Normal search completed",
                extra={
                    "results_count": len(results),
                    "search_limit": self.config.search_limit
                }
            )
            
            return results
            
        except Exception as e:
            self.logger.error(
                f"Error in normal search: {str(e)}",
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
    
    def _search_with_selection(
        self,
        query_embedding: List[float],
        user_query: str,
        partition_names: Optional[List[str]] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs search with document selection:
        1. Selects relevant documents using LLM
        2. Searches only within selected documents (using partitions)
        
        Args:
            query_embedding: Embedding vector of the query.
            user_query: User query text.
            partition_names: List of partition names to search in (will be overridden by selected documents).
            filter_expr: Optional filter expression.
            
        Returns:
            List[Dict[str, Any]]: Search results from selected documents.
        """
        self.logger.debug("Executing search with document selection")
        
        try:
            # Step 1: Select relevant documents using LLM
            self.logger.info("Step 1: Selecting relevant documents with LLM")
            selected_file_ids = self.document_selector.run(user_query=user_query)
            
            if not selected_file_ids:
                self.logger.warning(
                    "No documents selected by LLM, returning empty results",
                    extra={"user_query": user_query}
                )
                return []
            
            self.logger.info(
                "Documents selected",
                extra={
                    "selected_count": len(selected_file_ids),
                    "selected_file_ids": selected_file_ids
                }
            )
            
            # Step 2: Search in selected documents using partitions
            # In Milvus, each document is typically stored in a partition named by file_id
            self.logger.info("Step 2: Searching in selected documents")
            
            # Connect to Milvus
            self.searcher.connect()
            
            # Search in each selected document partition
            all_results = []
            for file_id in selected_file_ids:
                try:
                    # Search in this document's partition
                    partition_results = self.searcher.search_by_partition(
                        query_embedding=query_embedding,
                        partition_name=file_id,
                        limit=self.config.search_limit
                    )
                    all_results.extend(partition_results)
                    self.logger.debug(
                        f"Found {len(partition_results)} results in partition {file_id}"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Error searching in partition {file_id}: {str(e)}",
                        extra={"file_id": file_id, "error_type": type(e).__name__}
                    )
                    # Continue with other partitions
                    continue
            
            # Sort by score (descending) and limit results
            all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            final_results = all_results[:self.config.search_limit]
            
            self.logger.info(
                "Search with selection completed",
                extra={
                    "selected_documents": len(selected_file_ids),
                    "total_results": len(all_results),
                    "final_results": len(final_results)
                }
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(
                f"Error in search with selection: {str(e)}",
                extra={
                    "user_query": user_query,
                    "error_type": type(e).__name__
                },
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
        """Closes connections with Milvus and document selector."""
        self.logger.info("Closing Search Pipeline connections")
        try:
            if self.document_selector is not None:
                self.document_selector.close()
            # Note: searcher.disconnect() is called after each search
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

