"""Tests de split_sources: separar y validar el bloque <sources> que añade el LLM."""

from src.chatbot.rag_chatbot import split_sources

RESULTS = [
    {"file_name": "principito.pdf", "pages": "*9* *10*", "text": "a"},
    {"file_name": "principito.pdf", "pages": "*11*", "text": "b"},
    {"file_name": "cv_candidate_1.pdf", "pages": "", "text": "c"},
]


def test_extracts_block_and_removes_it_from_answer() -> None:
    raw = (
        "El principito vivía en el asteroide B 612.\n"
        "Esta respuesta se ha compuesto gracias al documento principito.pdf, páginas 10 y 9.\n"
        '<sources>[{"file_name": "principito.pdf", "pages": ["10", "9"]}]</sources>'
    )
    answer, sources = split_sources(raw, RESULTS)
    assert "<sources>" not in answer
    assert answer.endswith("páginas 10 y 9.")
    assert sources == [{"file_name": "principito.pdf", "pages": ["9", "10"]}]


def test_drops_invented_documents_and_pages() -> None:
    raw = (
        "Respuesta.\n<sources>["
        '{"file_name": "principito.pdf", "pages": ["9", "99"]},'
        '{"file_name": "inventado.pdf", "pages": ["1"]}'
        "]</sources>"
    )
    _, sources = split_sources(raw, RESULTS)
    assert sources == [{"file_name": "principito.pdf", "pages": ["9"]}]


def test_accepts_int_pages_and_documents_without_pages() -> None:
    raw = (
        'Respuesta.<sources>[{"file_name": "principito.pdf", "pages": [11]},'
        '{"file_name": "cv_candidate_1.pdf", "pages": []}]</sources>'
    )
    _, sources = split_sources(raw, RESULTS)
    assert sources == [
        {"file_name": "principito.pdf", "pages": ["11"]},
        {"file_name": "cv_candidate_1.pdf", "pages": []},
    ]


def test_merges_duplicate_documents() -> None:
    raw = (
        'R.<sources>[{"file_name": "principito.pdf", "pages": ["10"]},'
        '{"file_name": "principito.pdf", "pages": ["9", "10"]}]</sources>'
    )
    _, sources = split_sources(raw, RESULTS)
    assert sources == [{"file_name": "principito.pdf", "pages": ["9", "10"]}]


def test_missing_block_returns_answer_without_sources() -> None:
    answer, sources = split_sources("  Sin bloque.  ", RESULTS)
    assert answer == "Sin bloque."
    assert sources == []


def test_malformed_block_is_removed_and_ignored() -> None:
    answer, sources = split_sources("Respuesta.\n<sources>[{roto</sources>", RESULTS)
    assert answer == "Respuesta."
    assert sources == []
