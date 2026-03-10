"""
Module for generating markdown with document information
"""

from typing import List, Union
from src.retrieval.models import DocumentSummary
from src.utils import get_logger


class MarkdownGenerator:
    """
    Class for generating markdown with document information.
    Creates readable format with emojis and clear structure.
    """

    def __init__(self):
        """Initializes markdown generator."""
        self.logger = get_logger(__name__)

    def generate_document_markdown(
        self, summary: Union[DocumentSummary, dict]
    ) -> str:
        """
        Generates markdown for a single document.

        Args:
            summary: DocumentSummary or dict with document information (Milvus fields).

        Returns:
            str: Formatted markdown of the document.
        """
        # Support both DocumentSummary and dict (e.g. from get_all_summaries)
        if isinstance(summary, dict):
            file_id = summary.get("file_id", "")
            file_name = summary.get("file_name", "")
            file_type = str(summary.get("file_type", "")).upper()
            pages = summary.get("pages", "0")
            chapters = summary.get("chapters", "0")
            full_images = summary.get("full_images", "0")
            description = summary.get("text", "")
        else:
            file_id = summary.file_id
            file_name = summary.file_name
            file_type = summary.file_type.upper()
            pages = summary.pages
            chapters = summary.chapters
            full_images = summary.full_images
            description = summary.text

        # Generate markdown
        markdown = f"""## 📄 {file_name}

- **ID:** `{file_id}`  
- **Tipo:** {file_type}  
- **Páginas:** {pages}  
- **Capítulos:** {chapters}  
- **Imágenes:** {full_images}  

**Descripción:**  
{description}
"""
        
        return markdown

    def generate_all_documents_markdown(
        self, summaries: List[Union[DocumentSummary, dict]]
    ) -> str:
        """
        Generates markdown for all documents.

        Args:
            summaries: List of DocumentSummary objects.

        Returns:
            str: Complete markdown with all documents.
        """
        self.logger.info(
            f"Generating markdown for {len(summaries)} documents",
            extra={"count": len(summaries)}
        )

        if not summaries:
            return "# 📚 Biblioteca de Documentos\n\nNo hay documentos disponibles."

        # General header
        markdown_parts = [
            "# 📚 Biblioteca de Documentos\n",
            f"Total de documentos: **{len(summaries)}**\n",
            "---\n"
        ]

        # Add each document
        for summary in summaries:
            document_markdown = self.generate_document_markdown(summary)
            markdown_parts.append(document_markdown)
            markdown_parts.append("---\n")

        # Join all parts
        full_markdown = "\n".join(markdown_parts)
        
        self.logger.info("Markdown generated successfully")
        
        return full_markdown

