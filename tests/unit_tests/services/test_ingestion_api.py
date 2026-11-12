"""Tests del API de ingesta con IngestionPipeline mockeado."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from services.ingestion.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unsupported_extension(client: TestClient) -> None:
    files = {"file": ("doc.xyz", b"content", "application/octet-stream")}
    data = {"collection_name": "c1", "extract_images": "false"}
    response = client.post("/ingest", files=files, data=data)
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


@patch("services.ingestion.routes.IngestionPipeline")
def test_ingest_success(mock_pipeline_cls: MagicMock, client: TestClient) -> None:
    mock_pipeline = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline
    mock_pipeline.__enter__.return_value = mock_pipeline
    mock_pipeline.__exit__.return_value = None
    mock_pipeline.process_single_file.return_value = (
        True,
        "indexed",
        {"file_id": "abc", "file_name": "x.pdf", "file_path": "/tmp/x.pdf"},
    )

    files = {"file": ("x.pdf", b"%PDF-1.4 test content", "application/pdf")}
    data = {
        "collection_name": "my_col",
        "extract_images": "true",
        "job_id": "job-99",
    }
    response = client.post("/ingest", files=files, data=data)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["message"] == "indexed"
    assert body["collection_name"] == "my_col"
    assert body["result_info"]["file_id"] == "abc"

    mock_pipeline.process_single_file.assert_called_once()
    call_kw = mock_pipeline.process_single_file.call_args[1]
    assert call_kw["extract_process_images"] is True
    assert call_kw["job_id"] == "job-99"
    path = call_kw["file_path"]
    assert path.endswith("x.pdf")
