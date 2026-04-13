"""
CV Recruitment: Document-selection search (via retrieval API)
=============================================================

Calls the HTTP search service (POST /search) with `with_selection`.

Requirement: retrieval service accessible (default http://localhost:8007).
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.utils import get_logger

logger = get_logger(__name__)

RETRIEVAL_BASE_URL = "http://localhost:8007"
SEARCH_TYPE = "with_selection"
SEARCH_LIMIT = 10

COLLECTION_NAME = "cv_recruitment"
JOB_ID = "retrieval_example_cv"


def run_search_queries() -> None:
    print("=" * 80)
    print("CV RECRUITMENT: SEARCH WITH DOCUMENT SELECTION (retrieval API)")
    print("=" * 80)
    print(f"Service: {RETRIEVAL_BASE_URL}")
    print(f"Strategy: {SEARCH_TYPE}")
    print("\n" + "=" * 80 + "\n")

    queries = [
        "Evaluate the best candidates based on the technical requirements of the job proposal, "
        "give each a score from 0 to 10, and rank them from highest to lowest score.",
    ]

    try:
        r = httpx.get(f"{RETRIEVAL_BASE_URL}/health", timeout=10.0)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ Cannot reach the retrieval service: {e}")
        return

    with httpx.Client(timeout=120.0) as client:
        for i, query in enumerate(queries, 1):
            print(f"\n{'─' * 80}")
            print(f"QUERY {i}: {query}")
            print("─" * 80)
            try:
                payload = {
                    "query": query,
                    "collection_name": COLLECTION_NAME,
                    "search_type": SEARCH_TYPE,
                    "search_limit": SEARCH_LIMIT,
                    "job_id": f"{JOB_ID}.{i}",
                }
                resp = client.post(f"{RETRIEVAL_BASE_URL}/search", json=payload)
                if resp.status_code != 200:
                    print(f"\n✗ HTTP {resp.status_code}: {resp.text}")
                    continue
                data = resp.json()
                results = data.get("results") or []

                if results:
                    print(f"\n✓ Found {len(results)} result(s):\n")
                    for j, result in enumerate(results[:3], 1):
                        print(f"{j}. Document: {result.get('file_name', 'N/A')}")
                        print(f"   Score: {result.get('score', 0):.4f}")
                        print(f"   Pages: {result.get('pages', 'N/A')}")
                        print(f"   Text: {result.get('text', '')[:200]}...")
                        print()
                else:
                    print("\n✗ No results found")

            except Exception as e:
                logger.error("Error on query %s: %s", i, e, exc_info=True)
                print(f"\n✗ Error: {e}")

    print("\n" + "=" * 80)
    print("SEARCH COMPLETE")
    print("=" * 80)


def main() -> None:
    print("\n🚀 Search via retrieval API - CV Recruitment Example\n")
    try:
        run_search_queries()
    except Exception as e:
        logger.error("Error running the example: %s", e, exc_info=True)
        print(f"\n❌ Error: {e}")
        print("\nMake sure the retrieval service is running and the documents are indexed.")


if __name__ == "__main__":
    main()