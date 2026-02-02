"""
Document Selector + Metadata Search Strategy
Selects relevant documents using LLM, then searches with metadata filters
"""
from typing import List, Dict, Any, Optional
from .base import SearchStrategy
from src.search.milvus import MilvusSearcher
from src.search.document_selection import DocumentSelector
from src.search.metadata import MetadataSelector


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
        
        # Initialize MilvusSearcher for document search (misma colección, partición 'documents')
        self.searcher = MilvusSearcher(
            db_name=config.milvus.dbname,
            collection_name=config.collection_name,
            alias=config.milvus.alias
        )
        
        # Initialize DocumentSelector (misma colección, partición 'summaries')
        self.document_selector = DocumentSelector(
            dbname=config.milvus.dbname,
            collection_name=config.collection_name,
            text_model=config.text_model,
            uri=config.milvus.uri,
            token=config.milvus.token,
            host=config.milvus.host,
            port=config.milvus.port
        )
        
        # Initialize MetadataSelector (misma colección, partición 'summaries')
        self.metadata_selector = MetadataSelector(
            dbname=config.milvus.dbname,
            collection_name=config.collection_name,
            text_model=config.text_model,
            max_tokens=config.chooser_max_tokens,
            temperature=config.chooser_temperature,
            uri=config.milvus.uri,
            token=config.milvus.token,
            host=config.milvus.host,
            port=config.milvus.port
        )
        
        self.logger.info(
            "DocumentSelectorMetadataSearchStrategy initialized",
            extra={
                "collection_name": config.collection_name,
                "partition_documents": config.PARTITION_DOCUMENTS,
                "partition_summaries": config.PARTITION_SUMMARIES,
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
            filter_expr: DEPRECATED - Metadata filters are now automatically extracted from user_query.
                        If provided, will be combined with auto-generated filters.
            
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
                "has_manual_filter_expr": filter_expr is not None
            }
        )
        
        try:
            # Step 1: Select relevant documents using LLM
            self.logger.info("Step 1: Selecting relevant documents with LLM")
            selected_file_ids = self.document_selector.run(user_query=user_query)
            
            if not selected_file_ids:
                #meter en una funcion y devolver que el sistema no detecto ningun documento relevante?
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
            
            # Step 2: Extract metadata and build filters for each document
            self.logger.info("Step 2: Extracting metadata and building filters")
            document_filters = self.metadata_selector.run(
                user_query=user_query,
                selected_file_ids=selected_file_ids
            )
            
            self.logger.info(
                "Metadata filters created",
                extra={
                    "documents_with_filters": len(document_filters),
                    "file_ids": [df.id for df in document_filters]
                }
            )
            
            # Step 3: Search with individual filters for each document
            self.logger.info("Step 3: Searching with individual filters per document")
            
            # Connect to Milvus
            self.searcher.connect()
            
            # Collect all results from individual searches
            all_results = []
            
            # If we have document filters, search with them
            if document_filters:
                # Search for each document with its specific filter
                for doc_filter_info in document_filters:
                    file_id = doc_filter_info.id
                    doc_filter = doc_filter_info.expresion_milvus
                    
                    try:
                        # If user provided manual filter_expr, combine it
                        if filter_expr:
                            doc_filter = f"({doc_filter}) and ({filter_expr})"
                        
                        self.logger.debug(
                            f"Searching document {file_id} with filter: {doc_filter}"
                        )
                        
                        # Search with this document's filter in the 'documents' partition
                        doc_results = self.searcher.search(
                            query_embedding=query_embedding,
                            limit=self.config.search_limit,  # Limit per document
                            partition_names=[self.config.PARTITION_DOCUMENTS],
                            filter_expr=doc_filter
                        )
                        
                        all_results.extend(doc_results)
                        self.logger.debug(
                            f"Found {len(doc_results)} results for document {file_id}"
                        )
                        
                    except Exception as e:
                        self.logger.warning(
                            f"Error searching document {file_id}: {str(e)}",
                            extra={"file_id": file_id, "error_type": type(e).__name__}
                        )
                        # Continue with other documents
                        continue
                
                # Also search documents without metadata filters (if any)
                documents_with_filters = {df.id for df in document_filters}
                documents_without_filters = [
                    fid for fid in selected_file_ids 
                    if fid not in documents_with_filters
                ]
                
                if documents_without_filters:
                    self.logger.info(
                        f"Searching {len(documents_without_filters)} documents without metadata filters"
                    )
                    # Simple filter for documents without metadata
                    for file_id in documents_without_filters:
                        try:
                            doc_filter = f'file_id == "{file_id}"'
                            
                            # If user provided manual filter_expr, combine it
                            if filter_expr:
                                doc_filter = f"({doc_filter}) and ({filter_expr})"
                            
                            doc_results = self.searcher.search(
                                query_embedding=query_embedding,
                                limit=self.config.search_limit,
                                partition_names=[self.config.PARTITION_DOCUMENTS],
                                filter_expr=doc_filter
                            )
                            all_results.extend(doc_results)
                        except Exception as e:
                            self.logger.warning(
                                f"Error searching document {file_id}: {str(e)}"
                            )
                            continue
            else:
                # No metadata filters, search all selected documents with simple filter
                self.logger.info("No metadata filters, searching all selected documents")
                file_ids_filter = " or ".join([f'file_id == "{fid}"' for fid in selected_file_ids])
                
                # If user provided manual filter_expr, combine it
                if filter_expr:
                    combined_filter = f"({file_ids_filter}) and ({filter_expr})"
                else:
                    combined_filter = file_ids_filter
                
                all_results = self.searcher.search(
                    query_embedding=query_embedding,
                    limit=self.config.search_limit,
                    partition_names=[self.config.PARTITION_DOCUMENTS],
                    filter_expr=combined_filter
                )
            
            # Sort all results by score (descending) and limit to search_limit
            all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            results = all_results[:self.config.search_limit]
            
            self.logger.info(
                "Search with selection and metadata completed",
                extra={
                    "selected_documents": len(selected_file_ids),
                    "documents_with_filters": len(document_filters),
                    "total_results_found": len(all_results),
                    "final_results": len(results),
                    "manual_filter_applied": filter_expr is not None
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
            if self.metadata_selector is not None:
                self.metadata_selector.close()
            # Note: searcher.disconnect() is called after each search
            self.logger.info("DocumentSelectorMetadataSearchStrategy closed successfully")
        except Exception as e:
            self.logger.error(
                f"Error closing DocumentSelectorMetadataSearchStrategy: {str(e)}",
                extra={"error_type": type(e).__name__},
                exc_info=True
            )