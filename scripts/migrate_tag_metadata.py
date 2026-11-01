"""
Migra `pages` y `chapters` de la partición `documents` al formato con asteriscos.

    "VI,VII"  ->  "*VI* *VII*"
    "12"      ->  "*12*"

Necesario para que funcione el filtro `chapters like "%*VII*%"` (requiere Milvus >= 2.4)
sobre documentos subidos antes del cambio, sin tener que volver a subirlos.

Uso (desde la raíz del repo, con el venv activado):
    python scripts/migrate_tag_metadata.py                       # dry-run, todas las colecciones
    python scripts/migrate_tag_metadata.py --collection X        # dry-run, una colección
    python scripts/migrate_tag_metadata.py --apply               # aplica los cambios

Con --apply, por cada colección:
  1. Guarda un JSON con las filas originales (incluidos embeddings) en --backup-dir.
  2. Inserta las filas transformadas (las colecciones usan auto_id, así que reciben ids nuevos).
  3. Solo después borra las filas antiguas por id.
  4. Comprueba que el número de filas no ha cambiado.

Es idempotente: las filas que ya tienen asteriscos se ignoran.
Los resúmenes (partición `summaries`) no se tocan.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pymilvus import Collection, connections, utility

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.milvus.metadata_tags import is_tagged, parse_tagged, to_tagged  # noqa: E402

TAGGED_FIELDS = ("pages", "chapters")
PARTITION = "documents"
DEFAULT_BACKUP_DIR = Path.home() / "dev" / "backups" / f"rag-project-milvus-{datetime.now():%Y-%m-%d}"


def query_all(collection: Collection, pk: str) -> list[dict]:
    """Lee todas las filas de la partición, con todos los campos."""
    rows: list[dict] = []
    iterator = collection.query_iterator(
        expr=f"{pk} >= 0",
        output_fields=["*"],
        partition_names=[PARTITION],
        batch_size=500,
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            rows.extend(dict(r) for r in batch)
    finally:
        iterator.close()
    return rows


def count_rows(collection: Collection, pk: str) -> int:
    res = collection.query(expr=f"{pk} >= 0", output_fields=["count(*)"], partition_names=[PARTITION])
    return int(res[0]["count(*)"])


def transform(row: dict) -> dict | None:
    """Devuelve la fila con pages/chapters etiquetados, o None si no hay nada que cambiar."""
    new = dict(row)
    changed = False
    for field in TAGGED_FIELDS:
        value = row.get(field) or ""
        if not value or is_tagged(value):
            continue
        new[field] = to_tagged(parse_tagged(value))
        changed = changed or new[field] != value
    return new if changed else None


def migrate_collection(name: str, apply: bool, backup_dir: Path) -> bool:
    collection = Collection(name)
    schema = collection.schema
    pk = schema.primary_field.name
    max_len = {f.name: f.params.get("max_length") for f in schema.fields if f.name in TAGGED_FIELDS}

    if PARTITION not in [p.name for p in collection.partitions]:
        print(f"[{name}] sin partición '{PARTITION}', se omite")
        return True

    collection.load()
    rows = query_all(collection, pk)
    before = count_rows(collection, pk)
    if len(rows) != before:
        print(f"[{name}] ERROR: leídas {len(rows)} filas pero count(*) = {before}. No se toca nada.")
        return False

    changes = [(row, new) for row in rows if (new := transform(row)) is not None]
    print(f"\n[{name}] filas: {before} | a migrar: {len(changes)} | ya migradas/vacías: {before - len(changes)}")

    too_long = [
        (row[pk], f, len(new[f]))
        for row, new in changes
        for f in TAGGED_FIELDS
        if max_len.get(f) and len(new[f]) > max_len[f]
    ]
    if too_long:
        print(f"[{name}] ERROR: valores más largos que el max_length del esquema: {too_long[:5]}. No se toca nada.")
        return False

    for row, new in changes[:5]:
        print(f"   pages {row['pages']!r:>12} -> {new['pages']!r:<16} chapters {row['chapters']!r} -> {new['chapters']!r}")
    if len(changes) > 5:
        print(f"   ... y {len(changes) - 5} más")

    if not changes or not apply:
        return True

    # 1. Copia de seguridad de las filas originales
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_file = backup_dir / f"{name}_{PARTITION}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(backup_file, "w", encoding="utf-8") as f:
        json.dump([row for row, _ in changes], f, ensure_ascii=False, default=float)
    print(f"[{name}] copia de seguridad: {backup_file}")

    # 2. Insertar las filas nuevas (sin pk: auto_id) y 3. borrar las antiguas
    old_ids = [row[pk] for row, _ in changes]
    new_rows = [{k: v for k, v in new.items() if k != pk} for _, new in changes]
    insert_result = collection.insert(new_rows, partition_name=PARTITION)
    collection.flush()
    print(f"[{name}] insertadas {insert_result.insert_count} filas nuevas")

    collection.delete(expr=f"{pk} in {old_ids}", partition_name=PARTITION)
    collection.flush()

    # 4. Verificar
    after = count_rows(collection, pk)
    remaining_old = collection.query(expr=f"{pk} in {old_ids}", output_fields=[pk], partition_names=[PARTITION])
    pending = sum(1 for r in query_all(collection, pk) if transform(r) is not None)
    ok = after == before and not remaining_old and pending == 0
    print(
        f"[{name}] {'OK' if ok else 'ERROR'}: filas antes {before}, después {after}, "
        f"antiguas sin borrar {len(remaining_old)}, sin migrar {pending}"
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", action="append", help="Colección a migrar (se puede repetir). Por defecto, todas.")
    parser.add_argument("--apply", action="store_true", help="Aplicar los cambios. Sin esta opción solo muestra lo que haría.")
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR, help=f"Carpeta para el JSON de las filas originales (por defecto {DEFAULT_BACKUP_DIR})")
    args = parser.parse_args()

    load_dotenv(project_root / ".env")
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")
    db_name = os.getenv("MILVUS_DB_NAME", "default")
    connections.connect(host=host, port=port, db_name=db_name)
    print(f"Milvus {utility.get_server_version()} en {host}:{port}, base de datos '{db_name}'")
    print("MODO: " + ("APLICAR CAMBIOS" if args.apply else "DRY-RUN (no se modifica nada; usa --apply)"))

    names = args.collection or sorted(utility.list_collections())
    results = [migrate_collection(name, args.apply, args.backup_dir) for name in names]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
