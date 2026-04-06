"""Tests del API de retrieval con SearchPipeline mockeado."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from services.retrieval.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("services.retrieval.routes.SearchPipeline")
def test_search_returns_results(mock_pipeline_cls: MagicMock, client: TestClient) -> None:
    mock_pipeline = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline
    mock_pipeline.__enter__.return_value = mock_pipeline
    mock_pipeline.__exit__.return_value = None
    mock_pipeline.search.return_value = [
        {"text": "fragmento", "score": 0.95, "file_name": "doc.pdf", "pages": "1"}
    ]

    payload = {
        "query": "¿hola?",
        "collection_name": "test_collection",
        "search_type": "simple",
        "search_limit": 5,
    }
    response = client.post("/search", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert len(body["results"]) == 1
    assert body["results"][0]["text"] == "fragmento"
    mock_pipeline.search.assert_called_once_with(
        query="¿hola?",
        filter_expr=None,
        job_id=None,
    )
