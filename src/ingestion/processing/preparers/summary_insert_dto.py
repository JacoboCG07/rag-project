"""
DTO for summary insert data.
Defines the exact field contract according to the collection schema.
Same schema as milvus_insert_dto (file_id, file_type, file_name, text, etc.).
"""

from datetime import datetime
from typing import Dict, List

from .milvus_insert_dto import MilvusInsertDTO


def build_summary_insert_dict(
    *,
    summary: str,
    text_embedding: List[float],
    file_id: str,
    file_type: str,
    file_name: str,
    num_pages: int = 0,
    chapters: str = "",
    num_images: int = 0,
) -> MilvusInsertDTO:
    """
    Builds a valid dict for Milvus insertion of a document summary.
    Uses the same schema as milvus_insert_dto.

    Args:
        summary: Summary text.
        text_embedding: Summary embedding vector.
        file_id: File ID.
        file_type: File type (e.g. summary_pdf).
        file_name: File name (e.g. summary_document.pdf).
        num_pages: Number of pages in the document.
        chapters: Lista de capítulos con nombres (ej: "capitulo 1, capitulo 2, capitulo 3").
                  Vacío si no hay capítulos.
        num_images: Number of images in the document.

    Returns:
        Dict compatible with the Milvus schema.
    """
    date = datetime.now().date().strftime("%Y-%m-%d")
    pages = str(num_pages)
    chapters_str = chapters[:500] if chapters else ""  # schema max_length=500
    full_images = str(num_images)
    return {
        "file_id": file_id,
        "file_type": file_type,
        "file_name": file_name,
        "text": summary,
        "text_embedding": text_embedding,
        "pages": pages,
        "chapters": chapters_str,
        "image_number": "",
        "image_number_in_page": "",
        "full_images": full_images,
        "date": date,
    }
