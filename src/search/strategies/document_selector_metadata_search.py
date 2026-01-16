"""
Document Selector + Metadata Search Strategy
Selects relevant documents using LLM, then searches with metadata filters
"""
from typing import List, Dict, Any, Optional
from .base import SearchStrategy
from src.search.milvus import MilvusSearcher
from src.search.document_selection import DocumentSelector


class DocumentSelectorMetadataSearchStrategy(SearchStrategy):
    """
    Document selector with metadata search strategy:
    1. Selects relevant documents using LLM
    2. Searches within selected documents AND applies metadata filters
    
    This strategy combines document selection with metadata filtering
    for more precise search results.
    """
    
    def __init__(self, config):
        """
        Initialize the document selector + metadata search strategy.
        
        Args:
            config: SearchPipelineConfig with configuration parameters.
        """
        super().__init__(config)
        
        # Validate text_model is present
        if config.text_model is None:
            raise ValueError(
                "text_model is required for DocumentSelectorMetadataSearchStrategy. "
                "Please configure it in SearchPipelineConfig."
            )
        
        # Initialize MilvusSearcher for document search
        self.searcher = MilvusSearcher(
            db_name=config.milvus.dbname,
            collection_name=config.collection_name_documents,
            alias=config.milvus.alias
        )
        
        # Initialize DocumentSelector
        self.document_selector = DocumentSelector(
            dbname=config.milvus.dbname,
            collection_name=config.collection_name_summaries,
            text_model=config.text_model,
            uri=config.milvus.uri,
            token=config.milvus.token,
            host=config.milvus.host,
            port=config.milvus.port
        )
        
        self.logger.info(
            "DocumentSelectorMetadataSearchStrategy initialized",
            extra={
                "collection_documents": config.collection_name_documents,
                "collection_summaries": config.collection_name_summaries,
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
        Performs search with document selection and metadata filters:
        1. Selects relevant documents using LLM
        2. Searches within selected documents applying metadata filters
        
        Args:
            query_embedding: Embedding vector of the query.
            user_query: User query text (REQUIRED for this strategy).
            partition_names: Ignored, will be overridden by selected documents.
            filter_expr: Optional metadata filter expression (e.g., 'pages == "1-5"').
            
        Returns:
            List[Dict[str, Any]]: Search results from selected documents with filters applied.
            
        Raises:
            ValueError: If user_query is not provided.
        """
        if user_query is None:
            raise ValueError(
                "user_query is required for DocumentSelectorMetadataSearchStrategy. "
                "Please provide the user's query text."
            )
        
        self.logger.info(
            "Executing search with document selection and metadata filters",
            extra={
                "user_query": user_query,
                "has_filter_expr": filter_expr is not None
            }
        )
        
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
            
            # Step 2: Build filter expression to search only in selected documents
            # Combine file_id filter with optional user metadata filter
            file_ids_filter = " or ".join([f'file_id == "{fid}"' for fid in selected_file_ids])
            
            if filter_expr:
                # Combine document selection with user's metadata filter
                combined_filter = f"({file_ids_filter}) and ({filter_expr})"
                self.logger.info(
                    "Combining document selection with metadata filter",
                    extra={"combined_filter": combined_filter}
                )
            else:
                combined_filter = file_ids_filter
            
            # Step 3: Search with combined filter
            self.logger.info("Step 2: Searching with document selection and metadata filters")
            
            # Connect to Milvus
            self.searcher.connect()
            
            # Perform search with combined filter
            results = self.searcher.search(
                query_embedding=query_embedding,
                limit=self.config.search_limit,
                filter_expr=combined_filter
            )
            
            self.logger.info(
                "Search with selection and metadata completed",
                extra={
                    "selected_documents": len(selected_file_ids),
                    "results_count": len(results),
                    "filter_applied": filter_expr is not None
                }
            )
            
            return results
            
        except Exception as e:
            self.logger.error(
                f"Error in search with selection and metadata: {str(e)}",
                extra={
                    "user_query": user_query,
                    "filter_expr": filter_expr,
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
        """
        Closes connections with Milvus and document selector.
        """
        self.logger.info("Closing DocumentSelectorMetadataSearchStrategy connections")
        try:
            if self.document_selector is not None:
                self.document_selector.close()
            # Note: searcher.disconnect() is called after each search
            self.logger.info("DocumentSelectorMetadataSearchStrategy closed successfully")
        except Exception as e:
            self.logger.error(
                f"Error closing DocumentSelectorMetadataSearchStrategy: {str(e)}",
                extra={"error_type": type(e).__name__},
                exc_info=True
            )

