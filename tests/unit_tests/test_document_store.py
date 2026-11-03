"""Tests del almacén de documentos originales."""

import pytest

from src.utils import document_store


def test_save_and_url(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DOCUMENTS_DIR", str(tmp_path))
    dest = document_store.save_document("col", "doc 1.pdf", b"%PDF")
    assert dest == tmp_path / "col" / "doc 1.pdf"
    assert dest.read_bytes() == b"%PDF"
    assert document_store.document_url("col", "doc 1.pdf", "3") == "/files/col/doc%201.pdf#page=3"
    assert document_store.document_url("col", "doc.pdf", "IV") == "/files/col/doc.pdf"


def test_save_rejects_traversal(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DOCUMENTS_DIR", str(tmp_path))
    dest = document_store.save_document("col", "../../evil.pdf", b"x")
    assert dest == tmp_path / "col" / "evil.pdf"
    with pytest.raises(ValueError):
        document_store.save_document("..", "a.pdf", b"x")
