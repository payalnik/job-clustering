#!/usr/bin/env python3
"""Apply schema.sql to the configured PostgreSQL database.

Prerequisites:
  * PostgreSQL 16+ with the `pgvector` extension installed at the server level.
  * A fresh database + user matching the settings in .env (or JOBPROC_DB_*).
    e.g. `createuser jobproc --pwprompt && createdb jobproc -O jobproc`

Idempotent — running twice is harmless. Prints a row-count summary on exit.

Usage:
    python -m scripts.init_db
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobproc import db
from jobproc.config import setup_logging


def main() -> int:
    setup_logging()
    db.init_schema()
    print(f"Listings:   {db.count_active():>8d} active")
    print(f"Embeddings: {db.count_embeddings():>8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
