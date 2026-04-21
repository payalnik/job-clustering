"""Encode un-embedded active listings via Gemini, write vectors to pgvector.

This runs after ingestion. It finds every listing that doesn't yet have
a row in `position_vectors`, builds the role-text, calls the embedder
in batches, and upserts the results. Empty/garbage texts are skipped
up-front so we never pay for embeddings we can't use.

At ~100 positions / 3s (after the sleep throttle) a fresh corpus of 50K
positions takes roughly 25 minutes. Progress is logged every 500.
"""

from __future__ import annotations

import logging
import time

from . import db
from .embedder import embed_texts
from .config import EMBED_BATCH, EMBED_SLEEP_SECONDS
from .texts import role_text

log = logging.getLogger("jobproc.encode")


def encode_new_positions(batch_size: int = EMBED_BATCH) -> int:
    """Embed every active listing that doesn't yet have a vector.

    Returns the number of successfully encoded positions. Idempotent:
    re-running it after a successful run is a no-op.
    """
    pending = db.get_unembedded_listings()
    if not pending:
        log.info("All active listings already have embeddings.")
        return 0

    n = len(pending)
    log.info("Encoding %d positions via Gemini (batches of %d)...", n, batch_size)
    t0 = time.time()
    encoded = 0

    for start in range(0, n, batch_size):
        batch = pending[start:start + batch_size]
        texts = [role_text(p["title"], p["description_raw"]) for p in batch]
        valid = [(p, t) for p, t in zip(batch, texts) if len(t.strip()) > 10]
        if not valid:
            continue
        rows, inputs = zip(*valid)
        vectors = embed_texts(list(inputs), label="encode")

        # Close + reopen connection on each batch. Long encode runs can leave
        # an idle PG connection in a broken-pipe state; the next db.cursor()
        # will transparently reconnect.
        db.close_connection()
        db.save_embeddings([
            (row["url_hash"], vectors[i]) for i, row in enumerate(rows)
        ])

        encoded += len(rows)
        if start + batch_size < n:
            time.sleep(EMBED_SLEEP_SECONDS)

        elapsed = time.time() - t0
        rate = encoded / elapsed if elapsed else 0
        eta = (n - encoded) / rate if rate else 0
        log.info("ENCODE %d/%d (%.0f/s, ~%.0fs left)", encoded, n, rate, eta)

    log.info("ENCODE done. %d positions in %.1fs", encoded, time.time() - t0)

    # Tidy up — vectors whose listing was deactivated this session.
    removed = db.delete_orphan_embeddings()
    if removed:
        log.info("Removed %d orphan embeddings", removed)
    return encoded
