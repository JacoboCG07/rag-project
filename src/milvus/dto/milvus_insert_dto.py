"""
DTO for Milvus insert data.
Defines the exact field contract according to the collection schema.
Prevents inserting fields not defined in the schema (e.g. source_id).
"""

from typing import TypedDict, List

SCHEMA_KEYS = {
    "file_id",
    "file_type",
    "file_name",
    "text",
    "text_embedding",
    "pages",
    "chapters",
    "image_number",
    "image_number_in_page",
    "full_images",
    "date",
}


class MilvusInsertDTO(TypedDict, total=False):
    """
    Fields allowed for Milvus insertion.
    Aligned with the schema in rag_schema.py.
    Note: 'id' is auto_id in the schema, not included.
    """

    file_id: str
    file_type: str
    file_name: str
    text: str
    text_embedding: List[float]
    pages: str
    chapters: str
    image_number: str
    image_number_in_page: str
    full_images: str
    date: str


def build_milvus_insert_dict(
    *,
    file_id: str,
    file_name: str,
    file_type: str,
    text: str,
    text_embedding: List[float],
    date: str,
    pages: str = "",
    chapters: str = "",
    image_number: str = "",
    image_number_in_page: str = "",
    full_images: str = "",
) -> MilvusInsertDTO:
    """
    Builds a valid dict for Milvus insertion.
    Only includes fields defined in the schema.

    Args:
        file_id: File ID.
        file_name: File name.
        file_type: File type (schema uses file_type).
        text: Chunk text.
        text_embedding: Embedding vector.
        date: Date in YYYY-MM-DD format.
        pages: Pages (optional).
        chapters: Chapters (optional).
        image_number: Image number (optional).
        image_number_in_page: Image in page (optional).
        full_images: Full images (optional).

    Returns:
        Dict compatible with the Milvus schema.
    """
    return {
        "file_id": file_id,
        "file_type": file_type,
        "file_name": file_name,
        "text": text,
        "text_embedding": text_embedding,
        "pages": pages,
        "chapters": chapters,
        "image_number": image_number,
        "image_number_in_page": image_number_in_page,
        "full_images": full_images,
        "date": date,
    }
