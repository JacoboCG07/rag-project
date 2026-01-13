"""
Module for generating markdown with document information
"""

from typing import List, Dict, Any
from src.utils import get_logger


class MarkdownGenerator:
    """
    Class for generating markdown with document information.
    Creates readable format with emojis and clear structure.
    """

    def __init__(self):
        """Initializes markdown generator."""
        self.logger = get_logger(__name__)

    def generate_document_markdown(self, summary: Dict[str, Any]) -> str:
        """
        Generates markdown for a single document.

        Args:
            summary: Dictionary with document information.
                Must contain: file_name, type_file, total_pages,
                total_chapters, total_num_image, text

        Returns:
            str: Formatted markdown of the document.
        """
        # Extract summary information
        file_id = summary.get("file_id", "unknown_id")
        file_name = summary.get("file_name", "unnamed_document")
        type_file = summary.get("type_file", "UNKNOWN").upper()
        total_pages = summary.get("total_pages", "0")
        total_chapters = summary.get("total_chapters", "0")
        total_num_image = summary.get("total_num_image", "0")
        description = summary.get("text", "No description available.")

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

    def generate_all_documents_markdown(self, summaries: List[Dict[str, Any]]) -> str:
        """
        Generates markdown for all documents.

        Args:
            summaries: List of dictionaries with document information.

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

