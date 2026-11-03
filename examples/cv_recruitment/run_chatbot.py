"""
CV Recruitment: RAG Chatbot (via chatbot API)
=============================================

Calls the HTTP chat service (POST /chat): retrieval with `with_selection` + LLM.

Requirement: chatbot service accessible (default http://localhost:8018).
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.utils import get_logger

logger = get_logger(__name__)

CHATBOT_BASE_URL = "http://localhost:8018"

MODEL = None  # None = modelo por defecto del proveedor (Azure: AZURE_OPENAI_CHAT_DEPLOYMENT_NAME)
SEARCH_TYPE = "with_selection"
SEARCH_LIMIT = 10

COLLECTION_NAME = "cv_recruitment"
JOB_ID = "chatbot_example_cv"


def main() -> None:
    print("\n" + "=" * 80)
    print("RAG CHATBOT (chatbot API) - CV Recruitment System")
    print("=" * 80)
    print(f"Service: {CHATBOT_BASE_URL}")
    print(f"Collection in POST /chat: {COLLECTION_NAME}")
    print(f"Strategy: {SEARCH_TYPE}")
    print("=" * 80 + "\n")

    try:
        r = httpx.get(f"{CHATBOT_BASE_URL}/health", timeout=10.0)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ Cannot reach the chatbot service: {e}")
        return

    queries = [
        "Evaluate the best candidates based on the technical requirements of the job proposal, "
        "give each a score from 0 to 10, and rank them from highest to lowest score.",
    ]

    with httpx.Client(timeout=180.0) as client:
        for i, query in enumerate(queries, 1):
            print(f"\n{'-' * 80}")
            print(f"QUESTION {i}: {query}")
            print("-" * 80)
            try:
                payload = {
                    "query": query,
                    "collection_name": COLLECTION_NAME,
                    "search_type": SEARCH_TYPE,
                    "search_limit": SEARCH_LIMIT,
                    "llm_model": MODEL,
                    "job_id": f"{JOB_ID}.{i}",
                }
                resp = client.post(f"{CHATBOT_BASE_URL}/chat", json=payload)
                if resp.status_code != 200:
                    print(f"\nHTTP {resp.status_code}: {resp.text}")
                    continue
                data = resp.json()
                answer = data.get("answer", "")
                print(f"\nAnswer:\n{answer}\n")
            except Exception as e:
                logger.error("Error on question %s: %s", i, e, exc_info=True)
                print(f"\nError: {e}\n")

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
