"""
Script de Subida de Documentos - Ejemplo 2: CVs y Propuesta de Trabajo
========================================================================

Indexa los PDF del ejemplo llamando al **servicio HTTP de ingesta** (multipart).

Requisito: servicio de ingesta accesible (por defecto http://localhost:8015).
Chunk y embedding los define el despliegue del servicio de ingesta.

Documentos a subir:
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
JOB_ID = "upload_documents_2"


def upload_documents() -> None:
    """Sube cada fichero del ejemplo vía POST /ingest."""

    print("=" * 80)
    print("SUBIDA DE DOCUMENTOS (API de ingesta) - Ejemplo 2: CVs")
    print("=" * 80)
    print(f"Servicio: {INGESTION_BASE_URL}")
    print("=" * 80)

    if EXTRACT_IMAGES:
        print("\n⚠️  EXTRACT_IMAGES está activado (True).")
        print("   Más gasto en OpenAI (descripciones de imágenes).\n")
        while True:
            respuesta = input("¿Quieres continuar? (s/n): ").strip().lower()
            if respuesta in ("s", "si", "y", "yes"):
                break
            if respuesta in ("n", "no"):
                print("Subida cancelada.")
                return
            print("Responde 's' para sí o 'n' para no.")

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
        print("⚠️  Archivos no encontrados en data/:")
        for f in missing:
            print(f"   - {f}")
        print("\nColoca los PDF en la carpeta 'data/' e intenta de nuevo.\n")
        return

    existing_files = [data_dir / f for f in expected_files]
    print(f"\n🚀 Enviando {len(existing_files)} documento(s) al servicio...\n")

    try:
        r = httpx.get(f"{INGESTION_BASE_URL}/health", timeout=10.0)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ No se puede contactar con el servicio de ingesta: {e}")
        print(f"   ¿Está levantado?  uvicorn services.ingestion.main:app --port 8015")
        print(f"   o  docker compose up -d ingestion")
        return

    successful = 0
    failed = 0

    with httpx.Client(timeout=INGEST_TIMEOUT) as client:
        for i, file_path in enumerate(existing_files, 1):
            print(f"\n{'─' * 80}")
            print(f"[{i}/{len(existing_files)}] {file_path.name}")
            print(f"Colección: {COLLECTION_NAME}")
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
                    print(f"✓ {file_path.name} procesado correctamente")
                    ri = body.get("result_info") or {}
                    print(f"  - File ID: {ri.get('file_id', 'N/A')}")
                    print(f"  - Mensaje: {body.get('message', '')}")
                    successful += 1
                else:
                    print(f"✗ Error: {body.get('message', body)}")
                    failed += 1

            except Exception as e:
                print(f"✗ Error: {e}")
                failed += 1

    print("\n" + "=" * 80)
    print("RESUMEN DE SUBIDA")
    print("=" * 80)
    print(f"✓ Exitosos: {successful}")
    print(f"✗ Fallidos: {failed}")

    if successful > 0:
        print("\n✅ Documentos listos para búsqueda.")
        print(f"   Colección: {COLLECTION_NAME}")
        print("   Ejecuta: python run_retrieval.py  |  python run_chatbot.py")


def main() -> None:
    print("\n📤 Subida vía API de ingesta - Ejemplo 2: CVs y propuesta\n")
    upload_documents()


if __name__ == "__main__":
    main()
