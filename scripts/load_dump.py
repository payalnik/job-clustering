#!/usr/bin/env python3
"""Restore job_listings + position_vectors from data/dump/*.copy.gz.

Skips the slow re-ingest + re-embed cycle. Run after `init_db.py` against
an empty schema. Idempotent: rows whose url_hash already exists are skipped.

Usage:
    python -m scripts.init_db          # create schema first
    python -m scripts.load_dump        # then load the dump
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobproc import db
from jobproc.config import DATA_DIR, setup_logging

DUMP_DIR = DATA_DIR / "dump"
LISTINGS_FILE = DUMP_DIR / "job_listings.copy.gz"
VECTORS_FILE = DUMP_DIR / "position_vectors.copy.gz"

LISTINGS_COLS = (
    "url, url_hash, title, company, description_raw, location, "
    "remote_onsite, seniority, posted_date, source_domain, "
    "scraped_at, last_refreshed, is_active"
)


def _copy_into(table: str, columns: str, path: Path) -> int:
    """Stream a gzipped COPY file into a staging table, then INSERT ... ON CONFLICT.

    Using a temp table lets us re-run the loader without unique-key errors.
    """
    conn = db.get_connection()
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE TEMP TABLE _staging (LIKE {table} INCLUDING DEFAULTS) "
            f"ON COMMIT DROP"
        )
        with gzip.open(path, "rb") as fh:
            cur.copy_expert(f"COPY _staging ({columns}) FROM STDIN", fh)
        cur.execute(
            f"INSERT INTO {table} ({columns}) "
            f"SELECT {columns} FROM _staging "
            f"ON CONFLICT DO NOTHING"
        )
        inserted = cur.rowcount
        cur.execute("DROP TABLE _staging")
    return inserted


def main() -> int:
    log = setup_logging()
    if not LISTINGS_FILE.exists() or not VECTORS_FILE.exists():
        print(f"Dump files missing under {DUMP_DIR}", file=sys.stderr)
        return 1

    # Listings first — position_vectors has a FK on url_hash.
    log.info("Loading %s", LISTINGS_FILE.name)
    n = _copy_into("job_listings", LISTINGS_COLS, LISTINGS_FILE)
    log.info("Inserted %d listings", n)

    log.info("Loading %s", VECTORS_FILE.name)
    n = _copy_into("position_vectors", "url_hash, role_vec", VECTORS_FILE)
    log.info("Inserted %d vectors", n)

    print(f"Listings:   {db.count_active():>8d} active")
    print(f"Embeddings: {db.count_embeddings():>8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
