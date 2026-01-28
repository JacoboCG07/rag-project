"""
Module for metadata selection
Executes: retrieve summaries -> generate markdown -> extract metadata -> build filters
"""

from typing import List
from llms.text import BaseTextModel
from src.search.common import SummaryRetriever, MarkdownGenerator
from src.search.models import DocumentFilter, DocumentFilterWithDetails, DocumentSummary, DocumentMetadata
from .extractor import MetadataExtractor
from .filter_builder import MetadataFilterBuilder
from src.utils import get_logger


class MetadataSelector:
    """
    Metadata selector with LLM.
    Executes steps: Summaries -> Markdown -> LLM extraction -> Filter building
    
    Returns a list of dictionaries with document IDs and their Milvus filter expressions.
    """

    def __init__(
        self,
        dbname: str,
        collection_name: str,
        text_model: BaseTextModel,
        max_tokens: int = 500,
        temperature: float = 0.2,
        uri: str = None,
        token: str = None,
        host: str = None,
        port: str = None
    ):
        """
        Initializes the metadata selector.

        Args:
            dbname: Database name.
            collection_name: Summaries collection name.
            text_model: LLM model for metadata extraction.
            max_tokens: Maximum tokens for LLM response.
            temperature: Temperature for LLM (0.0-1.0).
            uri: Connection URI (optional).
            token: Authentication token (optional).
            host: Milvus host (optional).
            port: Milvus port (optional).
        """
        self.logger = get_logger(__name__)
        
        # Components
        self.retriever = SummaryRetriever(
            dbname=dbname,
            collection_name=collection_name,
            uri=uri,
            token=token,
            host=host,
            port=port
        )
        
        self.markdown_generator = MarkdownGenerator()
        
        self.extractor = MetadataExtractor(
            text_model=text_model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        self.filter_builder = MetadataFilterBuilder()
        
        self.logger.info("MetadataSelector initialized")

    def run(
        self,
        user_query: str,
        selected_file_ids: List[str]
    ) -> List[DocumentFilter]:
        """
        Executes the metadata selection process.
        
        Args:
            user_query: User query.
            selected_file_ids: List of file_ids to extract metadata for.
            
        Returns:
            List[DocumentFilter]: List of document filters with ID and Milvus expression.
            
            Example:
            [
                DocumentFilter(
                    id="doc_001",
                    expresion_milvus='file_id == "doc_001" and pages in ["1","2","3"]'
                ),
                DocumentFilter(
                    id="doc_002",
                    expresion_milvus='file_id == "doc_002" and chapters in ["cap1"]'
                )
            ]
        """
        self.logger.info(
            f"Executing metadata selection: {user_query}",
            extra={"selected_file_ids": selected_file_ids}
        )
        
        # Step 1: Retrieve summaries of selected documents
        summaries = self.retriever.get_summaries_by_file_ids(selected_file_ids)
        self.logger.info(f"✓ Summaries retrieved: {len(summaries)}")
        
        # Step 2: Generate markdown
        markdown = self.markdown_generator.generate_all_documents_markdown(summaries)
        self.logger.info(f"✓ Markdown generated")
        
        # Step 3: Extract metadata with LLM
        metadata_dict = self.extractor.extract(
            user_query=user_query,
            markdown_documents=markdown,
            documents_info=summaries
        )
        self.logger.info(f"✓ Metadata extracted: {len(metadata_dict)} documents")
        
        # Step 4: Build filter expressions for each document
        result = []
        for file_id, metadata in metadata_dict.items():
            filter_expr = self.filter_builder.build_filter_for_document(
                file_id=file_id,
                metadata=metadata
            )
            result.append(DocumentFilter(
                id=file_id,
                expresion_milvus=filter_expr
            ))
        
        self.logger.info(f"✓ Filters built: {len(result)} documents")
        
        return result

    def run_with_details(
        self,
        user_query: str,
        selected_file_ids: List[str]
    ) -> List[DocumentFilterWithDetails]:
        """
        Executes metadata selection and returns complete information.
        
        Args:
            user_query: User query.
            selected_file_ids: List of file_ids to extract metadata for.
            
        Returns:
            List[DocumentFilterWithDetails]: List of documents with complete details including
                metadata and filter expressions.
        """
        self.logger.info(
            f"Executing metadata selection with details: {user_query}",
            extra={"selected_file_ids": selected_file_ids}
        )
        
        # Retrieve summaries
        summaries = self.retriever.get_summaries_by_file_ids(selected_file_ids)
        
        # Generate markdown
        markdown = self.markdown_generator.generate_all_documents_markdown(summaries)
        
        # Extract metadata
        metadata_dict = self.extractor.extract(
            user_query=user_query,
            markdown_documents=markdown,
            documents_info=summaries
        )
        
        # Build complete result with all details
        result = []
        for file_id, metadata in metadata_dict.items():
            filter_expr = self.filter_builder.build_filter_for_document(
                file_id=file_id,
                metadata=metadata
            )
            
            # Find original summary
            summary_dict = next((s for s in summaries if s.get("file_id") == file_id), None)
            
            if summary_dict:
                result.append(DocumentFilterWithDetails(
                    id=file_id,
                    expresion_milvus=filter_expr,
                    metadata=DocumentMetadata(**metadata),
                    summary=DocumentSummary(**summary_dict)
                ))
        
        return result

    def close(self):
        """Closes connection."""
        self.retriever.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

