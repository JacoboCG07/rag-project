"""Tests del API de chatbot con RAGChatbot mockeado."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from services.chatbot.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("services.chatbot.routes.RAGChatbot")
def test_chat_returns_answer(mock_bot_cls: MagicMock, client: TestClient) -> None:
    mock_bot = MagicMock()
    mock_bot_cls.return_value = mock_bot
    mock_bot.__enter__.return_value = mock_bot
    mock_bot.__exit__.return_value = None
    mock_bot.ask_with_sources.return_value = (
        "Respuesta generada.",
        [{"file_name": "principito.pdf", "pages": ["9", "10"]}],
    )

    payload = {
        "query": "¿Hola?",
        "collection_name": "test_collection",
        "search_type": "simple",
        "search_limit": 10,
        "llm_model": "gpt-4o-mini",
        "job_id": "job-1",
    }
    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "answer": "Respuesta generada.",
        "sources": [
            {
                "file_name": "principito.pdf",
                "pages": ["9", "10"],
                "url": "/files/test_collection/principito.pdf#page=9",
            }
        ],
    }
    mock_bot.ask_with_sources.assert_called_once_with(query="¿Hola?", job_id="job-1")
