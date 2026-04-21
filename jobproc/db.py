"""Minimal PostgreSQL wrapper for the ingestion + neighborhood pipeline.

Design notes:
- Thread-local connections. Each thread holds one connection for its lifetime;
  no pool. With modest concurrency (PARALLEL=10) this is simpler than pgbouncer
  and has zero wiring cost for the researcher.
- autocommit=True. Prevents "idle in transaction" states that block autovacuum
  and can leak connections when the process crashes.
- Every row returned as a dict (RealDictCursor) so downstream code never has
  to care about column order.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from .config import PG_DSN, SCHEMA_PATH
from .hashing import url_hash as _hash
from .filters import clean_ats_url, is_garbage_url

log = logging.getLogger("jobproc.db")

_local = threading.local()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection():
    """Return the thread's connection, creating (and registering pgvector) if needed."""
    conn = getattr(_local, "conn", None)
    if conn is not None and not conn.closed:
        return conn
    conn = psycopg2.connect(**PG_DSN)
    conn.autocommit = True
    register_vector(conn)
    _local.conn = conn
    return conn


def close_connection() -> None:
    """Close the thread's connection. Safe to call on every thread exit."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        pass
    _local.conn = None


def cursor():
    return get_connection().cursor(cursor_factory=psycopg2.extras.RealDictCursor)


# --- Schema management -------------------------------------------------------

def init_schema() -> None:
    """Apply schema.sql to the configured database (idempotent)."""
    sql = SCHEMA_PATH.read_text()
    with cursor() as cur:
        cur.execute(sql)
    log.info("Schema applied from %s", SCHEMA_PATH.name)


# --- Listings CRUD -----------------------------------------------------------

def upsert_listings(listings: Sequence[dict]) -> int:
    """Insert or refresh a batch of listings.

    Each dict may contain: url, title, company, description_raw, location,
    remote_onsite, posted_date, source_domain.

    Returns the number of newly inserted rows. Existing URLs get their
    `last_refreshed` timestamp bumped and any non-empty fields overwritten,
    but empty incoming fields never clobber existing data.
    """
    if not listings:
        return 0

    # Reject rows without meaningful content. An embedding derived from
    # pure boilerplate or an empty string is worse than noise.
    listings = [
        l for l in listings
        if l.get("url") and (
            l.get("title", "").strip() or len(l.get("description_raw", "")) > 50
        )
    ]
    if not listings:
        return 0

    for l in listings:
        l["url"] = clean_ats_url(l["url"])
    listings = [l for l in listings if not is_garbage_url(l["url"])]
    if not listings:
        return 0

    now = _now_iso()
    urls = [l["url"] for l in listings]
    with cursor() as cur:
        placeholders = ",".join(["%s"] * len(urls))
        cur.execute(
            f"SELECT url FROM job_listings WHERE url IN ({placeholders})", urls,
        )
        existing = {row["url"] for row in cur.fetchall()}

        new_count = 0
        for l in listings:
            url = l["url"]
            if not url:
                continue
            h = _hash(url)
            title = _scrub_nul(l.get("title", ""))
            company = _scrub_nul(l.get("company", ""))
            desc = _scrub_nul(l.get("description_raw", ""))
            location = _scrub_nul(l.get("location", "") or "")
            remote = _scrub_nul(l.get("remote_onsite", "") or "")
            posted = _scrub_nul(l.get("posted_date", "") or "")
            domain = _scrub_nul(l.get("source_domain", "ats"))

            if url in existing:
                cur.execute(
                    """
                    UPDATE job_listings SET
                        last_refreshed = %s,
                        title = CASE WHEN %s != '' THEN %s ELSE title END,
                        company = CASE WHEN %s != '' THEN %s ELSE company END,
                        description_raw = CASE WHEN %s != '' THEN %s ELSE description_raw END,
                        location = CASE WHEN %s != '' THEN %s ELSE location END,
                        remote_onsite = CASE WHEN %s != '' THEN %s ELSE remote_onsite END,
                        posted_date = CASE WHEN %s != '' THEN %s ELSE posted_date END,
                        is_active = TRUE
                    WHERE url = %s
                    """,
                    (
                        now,
                        title, title, company, company, desc, desc,
                        location, location, remote, remote, posted, posted, url,
                    ),
                )
            else:
                existing.add(url)
                cur.execute(
                    """
                    INSERT INTO job_listings
                        (url, url_hash, title, company, description_raw,
                         location, remote_onsite, posted_date, source_domain,
                         scraped_at, last_refreshed)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (url, h, title, company, desc, location, remote, posted,
                     domain, now, now),
                )
                new_count += 1
        return new_count


def touch_refreshed(urls: Iterable[str], ts: str | None = None) -> None:
    """Bump last_refreshed on every row in `urls`. Used to mark 'still live'."""
    urls = list(urls)
    if not urls:
        return
    ts = ts or _now_iso()
    with cursor() as cur:
        for i in range(0, len(urls), 500):
            chunk = urls[i:i + 500]
            placeholders = ",".join(["%s"] * len(chunk))
            cur.execute(
                f"UPDATE job_listings SET last_refreshed = %s "
                f"WHERE url IN ({placeholders})",
                [ts] + chunk,
            )


def expire_unrefreshed(companies: Iterable[str], cutoff: str) -> int:
    """Mark is_active=false on rows whose last_refreshed < cutoff, among `companies`."""
    companies = list(companies)
    if not companies:
        return 0
    total = 0
    with cursor() as cur:
        for i in range(0, len(companies), 200):
            chunk = companies[i:i + 200]
            placeholders = ",".join(["%s"] * len(chunk))
            cur.execute(
                f"UPDATE job_listings SET is_active = FALSE "
                f"WHERE is_active AND source_domain = 'ats' "
                f"AND company IN ({placeholders}) "
                f"AND last_refreshed < %s",
                chunk + [cutoff],
            )
            total += cur.rowcount
    return total


def deactivate_company(company: str) -> int:
    with cursor() as cur:
        cur.execute(
            "UPDATE job_listings SET is_active = FALSE "
            "WHERE is_active AND source_domain = 'ats' AND company = %s",
            (company,),
        )
        return cur.rowcount


def get_urls_with_content(urls: Sequence[str]) -> set[str]:
    """Return the subset of URLs that already have description_raw > 100 chars."""
    if not urls:
        return set()
    result: set[str] = set()
    with cursor() as cur:
        for i in range(0, len(urls), 500):
            chunk = urls[i:i + 500]
            placeholders = ",".join(["%s"] * len(chunk))
            cur.execute(
                f"SELECT url FROM job_listings WHERE url IN ({placeholders}) "
                f"AND description_raw IS NOT NULL AND length(description_raw) > 100",
                chunk,
            )
            result.update(r["url"] for r in cur.fetchall())
    return result


# --- Counts / health ---------------------------------------------------------

def count_active() -> int:
    with cursor() as cur:
        cur.execute("SELECT COUNT(*) AS c FROM job_listings WHERE is_active")
        return cur.fetchone()["c"]


def count_embeddings() -> int:
    with cursor() as cur:
        cur.execute("SELECT COUNT(*) AS c FROM position_vectors")
        return cur.fetchone()["c"]


# --- Embeddings --------------------------------------------------------------

def get_embedded_url_hashes() -> set[int]:
    with cursor() as cur:
        cur.execute("SELECT url_hash FROM position_vectors")
        return {r["url_hash"] for r in cur.fetchall()}


def save_embeddings(items: Sequence[tuple[int, Any]]) -> None:
    """Upsert (url_hash, 768-dim numpy array) pairs into position_vectors."""
    if not items:
        return
    with cursor() as cur:
        for url_h, vec in items:
            cur.execute(
                "INSERT INTO position_vectors (url_hash, role_vec) VALUES (%s, %s) "
                "ON CONFLICT (url_hash) DO UPDATE SET role_vec = EXCLUDED.role_vec",
                (url_h, vec),
            )


def delete_orphan_embeddings() -> int:
    """Remove vectors for listings that have been deactivated."""
    with cursor() as cur:
        cur.execute(
            "DELETE FROM position_vectors "
            "WHERE url_hash NOT IN (SELECT url_hash FROM job_listings WHERE is_active)"
        )
        return cur.rowcount


def load_embeddings_for_hashes(hashes: Sequence[int]) -> dict[int, Any]:
    """Fetch {url_hash: role_vec} for a specific set of hashes."""
    if not hashes:
        return {}
    result: dict[int, Any] = {}
    with cursor() as cur:
        for i in range(0, len(hashes), 1000):
            chunk = hashes[i:i + 1000]
            placeholders = ",".join(["%s"] * len(chunk))
            cur.execute(
                f"SELECT url_hash, role_vec FROM position_vectors "
                f"WHERE url_hash IN ({placeholders})",
                chunk,
            )
            for row in cur.fetchall():
                result[row["url_hash"]] = row["role_vec"]
    return result


# --- Un-embedded listings ---------------------------------------------------

def get_unembedded_listings(limit: int | None = None) -> list[dict]:
    """Return active listings that don't yet have a vector in position_vectors.

    Returns fields needed by the text builder: url, url_hash, title,
    description_raw.
    """
    sql = (
        "SELECT jl.url, jl.url_hash, jl.title, jl.description_raw "
        "FROM job_listings jl "
        "LEFT JOIN position_vectors pv ON jl.url_hash = pv.url_hash "
        "WHERE jl.is_active AND pv.url_hash IS NULL "
        "AND (length(jl.title) > 0 OR length(jl.description_raw) > 50)"
    )
    params: list[Any] = []
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)
    with cursor() as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


# --- Misc --------------------------------------------------------------------

def _scrub_nul(s: str) -> str:
    """PostgreSQL TEXT rejects NUL bytes — drop them. No visual impact."""
    return s.replace("\x00", "") if isinstance(s, str) else s
