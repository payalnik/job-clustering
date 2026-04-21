"""Gemini embedding API wrapper.

Uses `gemini-embedding-2-preview` at 768 dimensions, which is what the
deconfound transform was trained on. Do not change the model or
dimensionality without retraining.

Batch size is 100 (the API's documented max). After each batch we
sleep briefly to leave headroom for other Gemini traffic in the same
project.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .config import EMBED_BATCH, EMBED_SLEEP_SECONDS, get_gemini_key

log = logging.getLogger("jobproc.embed")

_MODEL = "gemini-embedding-2-preview"
_DIMS = 768
_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=get_gemini_key())
    return _client


def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    label: str = "",
) -> list[np.ndarray]:
    """Embed a list of texts, batching under the Gemini API limit.

    task_type should be RETRIEVAL_DOCUMENT for corpus texts (job listings)
    and RETRIEVAL_QUERY for user queries. Using asymmetric task types at
    indexing and query time measurably improves retrieval quality.

    Returns L2-normalized float32 vectors so cosine similarity reduces to
    a plain dot product downstream.
    """
    from google.genai import types

    client = _get_client()
    out: list[np.ndarray] = []
    n = len(texts)
    t0 = time.time()

    for start in range(0, n, EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        # Gemini returns a 400 for empty Parts; replace with a benign token.
        batch = [t if t.strip() and len(t.strip()) > 5 else "untitled position" for t in batch]

        for attempt in range(3):
            try:
                result = client.models.embed_content(
                    model=_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=_DIMS,
                        task_type=task_type,
                    ),
                )
                if len(result.embeddings) != len(batch):
                    raise RuntimeError(
                        f"Gemini returned {len(result.embeddings)} embeddings "
                        f"for {len(batch)} inputs"
                    )
                for emb in result.embeddings:
                    v = np.asarray(emb.values, dtype=np.float32)
                    norm = np.linalg.norm(v)
                    if norm > 0:
                        v = v / norm
                    out.append(v)
                break
            except Exception as e:
                if attempt == 2:
                    log.error("Gemini embed failed: %s", e)
                    raise
                wait = 1 + attempt * 2
                log.warning("Gemini retry %d after %ds: %s", attempt + 1, wait, e)
                time.sleep(wait)

        # Periodic progress log for long encoding runs.
        end = start + len(batch)
        if n > EMBED_BATCH and end % 500 < EMBED_BATCH:
            elapsed = time.time() - t0
            rate = end / elapsed if elapsed > 0 else 0
            eta = (n - end) / rate if rate > 0 else 0
            prefix = f"{label}: " if label else ""
            log.info("%s%d/%d (%.0f/s, ~%.0fs left)", prefix, end, n, rate, eta)

    return out
