"""Tests del formato con asteriscos para pages/chapters (src/milvus/metadata_tags.py)."""

import sys
from pathlib import Path

import pytest

_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.milvus.metadata_tags import is_tagged, like_any, parse_tagged, to_tagged
from src.retrieval.metadata.filter_builder import MetadataFilterBuilder
from src.ingestion.processing.preparers.document_preparer import DocumentPreparer


class TestToTagged:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (["VI", "VII"], "*VI* *VII*"),
            ([1, 2], "*1* *2*"),
            (("I",), "*I*"),
            (5, "*5*"),
            (0, "*0*"),
            ("VII", "*VII*"),
            (" IV ", "*IV*"),
            ("*VI* *VII*", "*VI* *VII*"),  # ya etiquetado: no se toca
            (None, ""),
            (False, ""),
            ("", ""),
            ([], ""),
            (["", " "], ""),
        ],
    )
    def test_to_tagged(self, value, expected):
        assert to_tagged(value) == expected


class TestParseTagged:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("*VI* *VII*", ["VI", "VII"]),
            ("*11*", ["11"]),
            ("*Capítulo 1* *Capítulo 2*", ["Capítulo 1", "Capítulo 2"]),
            ("VI,VII", ["VI", "VII"]),  # formato antiguo con comas
            ("12", ["12"]),
            ("", []),
            (None, []),
        ],
    )
    def test_parse_tagged(self, value, expected):
        assert parse_tagged(value) == expected

    def test_roundtrip(self):
        values = ["XIX", "XVIII", "XVII"]
        assert parse_tagged(to_tagged(values)) == values

    def test_is_tagged(self):
        assert is_tagged("*1* *2*")
        assert not is_tagged("1,2")
        assert not is_tagged(None)


class TestLikeAny:
    def test_single_value(self):
        assert like_any("chapters", ["VII"]) == 'chapters like "%*VII*%"'

    def test_several_values(self):
        assert like_any("pages", [1, 2]) == '(pages like "%*1*%" or pages like "%*2*%")'

    def test_empty(self):
        assert like_any("pages", []) == ""
        assert like_any("pages", None) == ""
        assert like_any("pages", ["", "*"]) == ""

    def test_removes_unsafe_chars_and_duplicates(self):
        assert like_any("chapters", ['V"II', "*VII*", "VII%"]) == 'chapters like "%*VII*%"'

    @pytest.mark.parametrize(
        "stored, search, matches",
        [
            ("*V*", "V", True),
            ("*IV* *V*", "V", True),
            ("*VI* *V*", "V", True),
            ("*VI* *VII*", "V", False),
            ("*XV*", "V", False),
            ("*VIII* *VII*", "VII", True),
            ("*XVII*", "VII", False),
            ("*1* *2*", "1", True),
            ("*11* *12*", "1", False),
        ],
    )
    def test_pattern_semantics(self, stored, search, matches):
        """Simula el LIKE "%*X*%" de Milvus: no debe haber coincidencias parciales."""
        expr = like_any("chapters", [search])
        needle = expr.split('"%')[1].split('%"')[0]
        assert (needle in stored) is matches


class TestFilterBuilder:
    def test_chapters_and_pages_use_like(self):
        expr = MetadataFilterBuilder().build_filter_for_document(
            "doc1", {"chapters": ["VII"], "pages": [12, 13], "file_type": "PDF"}
        )
        assert expr == (
            'file_id == "doc1" and (pages like "%*12*%" or pages like "%*13*%") '
            'and chapters like "%*VII*%" and file_type == ".pdf"'
        )


class TestDocumentPreparer:
    def test_pages_and_chapters_are_tagged(self):
        rows = DocumentPreparer.prepare(
            texts=["a", "b", "c"],
            embeddings=[[0.1], [0.2], [0.3]],
            file_metadata={"file_id": "f1", "file_name": "doc.pdf", "type_file": ".pdf"},
            chunks_metadata=[
                {"pages": [1, 2], "chapters": ["I", "II"]},
                {"pages": 5},  # imagen: página suelta, sin capítulos
                {},
            ],
        )
        assert [(r["pages"], r["chapters"]) for r in rows] == [
            ("*1* *2*", "*I* *II*"),
            ("*5*", ""),
            ("", ""),
        ]
