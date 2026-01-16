"""
Document Selector Search Strategy
First selects relevant documents using LLM, then searches within them
"""
from typing import List, Dict, Any, Optional
from .base import SearchStrategy
from src.search.milvus import MilvusSearcher
from src.search.document_selection import DocumentSelector


class DocumentSelectorSearchStrategy(SearchStrategy):
    """
    Document selector search strategy:
    1. Selects relevant documents using LLM
    2. Searches only within selected documents (using partitions)
    """
    
    def __init__(self, config):
        """
        Initialize the document selector search strategy.
        
        Args:
            config: SearchPipelineConfig with configuration parameters.
        """
        super().__init__(config)
        
        # Validate text_model is present
        if config.text_model is None:
            raise ValueError(
                "text_model is required for DocumentSelectorSearchStrategy. "
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
            "DocumentSelectorSearchStrategy initialized",
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
        Performs search with document selection:
        1. Selects relevant documents using LLM
        2. Searches only within selected documents
        
        Args:
            query_embedding: Embedding vector of the query.
            user_query: User query text (REQUIRED for this strategy).
            partition_names: Ignored, will be overridden by selected documents.
            filter_expr: Optional filter expression (not used in this strategy).
            
        Returns:
            List[Dict[str, Any]]: Search results from selected documents.
            
        Raises:
            ValueError: If user_query is not provided.
        """
        if user_query is None:
            raise ValueError(
                "user_query is required for DocumentSelectorSearchStrategy. "
                "Please provide the user's query text."
            )
        
        self.logger.info(
            "Executing search with document selection",
            extra={"user_query": user_query}
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
            
            # Step 2: Search in selected documents using partitions
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
        """
        Closes connections with Milvus and document selector.
        """
        self.logger.info("Closing DocumentSelectorSearchStrategy connections")
        try:
            if self.document_selector is not None:
                self.document_selector.close()
            # Note: searcher.disconnect() is called after each search
            self.logger.info("DocumentSelectorSearchStrategy closed successfully")
        except Exception as e:
            self.logger.error(
                f"Error closing DocumentSelectorSearchStrategy: {str(e)}",
                extra={"error_type": type(e).__name__},
                exc_info=True
            )

