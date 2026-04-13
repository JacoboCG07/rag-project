"""
Document Upload Script - CV Recruitment Example
================================================

Indexes the example PDFs by calling the **ingestion HTTP service** (multipart).

Requirement: ingestion service accessible (default http://localhost:8015).
Chunk size and embedding model are defined by the ingestion service deployment.

Documents to upload:
- data/job_proposal.pdf
- data/cv_candidate_1.pdf … cv_candidate_4.pdf
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import httpx

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

if sys.platform == "win32":
    import io as _io

    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

INGESTION_BASE_URL = "http://localhost:8015"
INGEST_TIMEOUT = 600.0
EXTRACT_IMAGES = False

COLLECTION_NAME = "cv_recruitment"
JOB_ID = "upload_documents_cv"


def upload_documents() -> None:
    """Uploads each example file via POST /ingest."""

    print("=" * 80)
    print("DOCUMENT UPLOAD (ingestion API) - CV Recruitment Example")
    print("=" * 80)
    print(f"Service: {INGESTION_BASE_URL}")
    print("=" * 80)

    if EXTRACT_IMAGES:
        print("\n⚠️  EXTRACT_IMAGES is enabled (True).")
        print("   This will increase OpenAI usage (image descriptions).\n")
        while True:
            answer = input("Do you want to continue? (y/n): ").strip().lower()
            if answer in ("y", "yes"):
                break
            if answer in ("n", "no"):
                print("Upload cancelled.")
                return
            print("Please answer 'y' for yes or 'n' for no.")

    data_dir = Path(__file__).parent / "data"
    expected_files = [
        "job_proposal.pdf",
        "cv_candidate_1.pdf",
        "cv_candidate_2.pdf",
        "cv_candidate_3.pdf",
        "cv_candidate_4.pdf",
    ]
    missing = [f for f in expected_files if not (data_dir / f).exists()]
    if missing:
        print("⚠️  Files not found in data/:")
        for f in missing:
            print(f"   - {f}")
        print("\nPlace the PDF files in the 'data/' folder and try again.\n")
        return

    existing_files = [data_dir / f for f in expected_files]
    print(f"\n🚀 Sending {len(existing_files)} document(s) to the service...\n")

    try:
        r = httpx.get(f"{INGESTION_BASE_URL}/health", timeout=10.0)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ Cannot reach the ingestion service: {e}")
        print(f"   Is it running?  uvicorn services.ingestion.main:app --port 8015")
        print(f"   or  docker compose up -d ingestion")
        return

    successful = 0
    failed = 0

    with httpx.Client(timeout=INGEST_TIMEOUT) as client:
        for i, file_path in enumerate(existing_files, 1):
            print(f"\n{'─' * 80}")
            print(f"[{i}/{len(existing_files)}] {file_path.name}")
            print(f"Collection: {COLLECTION_NAME}")
            print("─" * 80)

            extract_str = "true" if EXTRACT_IMAGES else "false"
            try:
                with open(file_path, "rb") as f:
                    file_content = f.read()
                files = {
                    "file": (file_path.name, io.BytesIO(file_content), "application/pdf"),
                }
                form = {
                    "collection_name": COLLECTION_NAME,
                    "extract_images": extract_str,
                    "job_id": f"{JOB_ID}.{i}",
                }
                resp = client.post(f"{INGESTION_BASE_URL}/ingest", files=files, data=form)
                try:
                    body = resp.json()
                except Exception:
                    body = {}

                if resp.status_code != 200:
                    print(f"✗ HTTP {resp.status_code}: {body.get('detail', resp.text)}")
                    failed += 1
                    continue

                if body.get("success"):
                    print(f"✓ {file_path.name} processed successfully")
                    ri = body.get("result_info") or {}
                    print(f"  - File ID: {ri.get('file_id', 'N/A')}")
                    print(f"  - Message: {body.get('message', '')}")
                    successful += 1
                else:
                    print(f"✗ Error: {body.get('message', body)}")
                    failed += 1

            except Exception as e:
                print(f"✗ Error: {e}")
                failed += 1

    print("\n" + "=" * 80)
    print("UPLOAD SUMMARY")
    print("=" * 80)
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")

    if successful > 0:
        print("\n✅ Documents ready for search.")
        print(f"   Collection: {COLLECTION_NAME}")
        print("   Next steps (from the repo root, one command per line):")
        print("     python examples/cv_recruitment/run_retrieval.py")
        print("     python examples/cv_recruitment/run_chatbot.py")


def main() -> None:
    print("\n📤 Upload via ingestion API - CV Recruitment Example\n")
    upload_documents()


if __name__ == "__main__":
    main()
