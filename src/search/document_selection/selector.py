"""
Module for document selection
Executes: retrieve summaries -> generate markdown -> select with LLM
"""

from typing import Optional, List, Dict, Any
from llms.text import BaseTextModel
from .retriever import SummaryRetriever
from .formatter import MarkdownGenerator
from .chooser import LLMDocumentChooser
from src.utils import get_logger


class DocumentSelector:
    """
    Document selector with LLM.
    Executes steps: Milvus -> Markdown -> LLM -> Selected IDs
    """

    def __init__(
        self,
        dbname: str,
        collection_name: str,
        text_model: BaseTextModel,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None
    ):
        """
        Initializes the selector.

        Args:
            dbname: Database name.
            collection_name: Summaries collection name.
            text_model: LLM model for selection.
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
        
        self.chooser = LLMDocumentChooser(
            text_model=text_model,
            max_tokens=500,
            temperature=0.2
        )
        
        self.logger.info("DocumentSelector initialized")

    def run(self, user_query: str) -> List[str]:
        """
        Executes the selection process.
        
        Args:
            user_query: User query.
            
        Returns:
            List[str]: List of selected file_ids.
        """
        self.logger.info(f"Executing selection: {user_query}")
        
        # Step 1: Retrieve summaries
        summaries = self.retriever.get_all_summaries()
        self.logger.info(f"✓ Summaries retrieved: {len(summaries)}")
        
        # Step 2: Generate markdown
        markdown = self.markdown_generator.generate_all_documents_markdown(summaries)
        self.logger.info(f"✓ Markdown generated")
        
        # Step 3: Select with LLM
        selected_ids = self.chooser.choose_documents(
            markdown_descriptions=markdown,
            user_query=user_query,
            summaries=summaries
        )
        self.logger.info(f"✓ Documents selected: {len(selected_ids)}")
        
        return selected_ids

    def run_with_details(self, user_query: str) -> List[Dict[str, Any]]:
        """
        Executes selection and returns complete information.
        
        Args:
            user_query: User query.
            
        Returns:
            List[Dict[str, Any]]: List of documents with complete details.
        """
        self.logger.info(f"Executing selection with details: {user_query}")
        
        # Step 1: Retrieve summaries
        summaries = self.retriever.get_all_summaries()
        
        # Step 2: Generate markdown
        markdown = self.markdown_generator.generate_all_documents_markdown(summaries)
        
        # Step 3: Select with details
        selected_documents = self.chooser.choose_documents_with_details(
            markdown_descriptions=markdown,
            user_query=user_query,
            summaries=summaries
        )
        
        return selected_documents

    def close(self):
        """Closes connection."""
        self.retriever.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

