"""
Module for generating markdown with document information
"""

from typing import List
from src.search.models import DocumentSummary
from src.utils import get_logger


class MarkdownGenerator:
    """
    Class for generating markdown with document information.
    Creates readable format with emojis and clear structure.
    """

    def __init__(self):
        """Initializes markdown generator."""
        self.logger = get_logger(__name__)

    def generate_document_markdown(self, summary: DocumentSummary) -> str:
        """
        Generates markdown for a single document.

        Args:
            summary: DocumentSummary with document information.

        Returns:
            str: Formatted markdown of the document.
        """
        # Extract summary information
        file_id = summary.file_id
        file_name = summary.file_name
        type_file = summary.type_file.upper()
        total_pages = summary.total_pages
        total_chapters = summary.total_chapters
        total_num_image = summary.total_num_image
        description = summary.text

        # Generate markdown
        markdown = f"""## 📄 {file_name}

- **ID:** `{file_id}`  
- **Tipo:** {type_file}  
- **Páginas:** {total_pages}  
- **Capítulos:** {total_chapters}  
- **Imágenes:** {total_num_image}  

**Descripción:**  
{description}
"""
        
        return markdown

    def generate_all_documents_markdown(self, summaries: List[DocumentSummary]) -> str:
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

