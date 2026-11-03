"""
Original documents store on disk, so cited sources can be opened from the UI.

Layout: {DOCUMENTS_DIR}/{collection_name}/{file_name}

The ingestion service writes a copy of each processed file; the chatbot service serves the
folder read-only under /files/{collection_name}/{file_name}.
"""

import os
from pathlib import Path
from urllib.parse import quote

DEFAULT_DOCUMENTS_DIR = "volumes/documents"


def documents_dir() -> Path:
    """Root folder of the store (env DOCUMENTS_DIR, default ./volumes/documents)."""
    return Path(os.getenv("DOCUMENTS_DIR", DEFAULT_DOCUMENTS_DIR))


def _safe_part(name: str) -> str:
    part = Path(name).name
    if not part or part in (".", ".."):
        raise ValueError(f"Invalid path component: {name!r}")
    return part


def save_document(collection_name: str, file_name: str, contents: bytes) -> Path:
    """Write (or overwrite) the original file of a document in the store."""
    folder = documents_dir() / _safe_part(collection_name)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / _safe_part(file_name)
    dest.write_bytes(contents)
    return dest


def document_url(collection_name: str, file_name: str, page: str | None = None) -> str:
    """Relative URL served by the chatbot service; `#page=N` makes browser PDF viewers jump to N."""
    url = f"/files/{quote(collection_name)}/{quote(file_name)}"
    if page and str(page).isdigit():
        url += f"#page={page}"
    return url
