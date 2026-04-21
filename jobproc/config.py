"""Central configuration — DB DSN, API keys, paths.

Every setting is read from (in order):
  1. An explicit environment variable
  2. The project-root `.env` file (key=value format)
  3. A documented default

This keeps secrets out of the source tree while still letting the
whole pipeline run from a single checkout with no extra wiring.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TARGET_COMPANIES_PATH = DATA_DIR / "target_companies.json"
DECONFOUND_TRANSFORM_PATH = DATA_DIR / "deconfound_transform.npz"
SCHEMA_PATH = PROJECT_ROOT / "schema.sql"


def _from_dotenv(name: str) -> str | None:
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    return None


def _get(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or _from_dotenv(name) or default


# --- PostgreSQL ---

PG_DSN: dict[str, str | int] = {
    "host": _get("JOBPROC_DB_HOST", "localhost") or "localhost",
    "port": int(_get("JOBPROC_DB_PORT", "5432") or 5432),
    "dbname": _get("JOBPROC_DB_NAME", "jobproc") or "jobproc",
    "user": _get("JOBPROC_DB_USER", "jobproc") or "jobproc",
    "password": _get("JOBPROC_DB_PASSWORD", "") or "",
}


# --- Gemini ---

def get_gemini_key() -> str:
    key = _get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Put it in .env or export it. "
            "Get one at https://aistudio.google.com/apikey."
        )
    return key


# --- Ingestion tuning ---

PARALLEL = int(_get("JOBPROC_PARALLEL", "10") or 10)
EMBED_BATCH = int(_get("JOBPROC_EMBED_BATCH", "100") or 100)
EMBED_SLEEP_SECONDS = float(_get("JOBPROC_EMBED_SLEEP", "3") or 3)

REQUEST_TIMEOUT = 15  # seconds, per HTTP call
USER_AGENT = "Mozilla/5.0 (compatible; JobProc/0.1; +https://github.com/)"


# --- Logging ---

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure a single stderr handler with a tidy format.

    Safe to call multiple times — later calls are a no-op.
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(handler)
    root.setLevel(level)
    # Quiet noisy libraries.
    for noisy in ("urllib3", "requests", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("jobproc")
