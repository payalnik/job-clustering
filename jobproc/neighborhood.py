"""Bottom-up neighborhood discovery for a role.

For a given center role (e.g. "Data Scientist"), we:

  1. Find every active listing whose normalized title matches the center.
  2. Compute the raw centroid of those listings' embeddings.
  3. Use pgvector to fetch the N nearest candidates to that centroid,
     across the full active corpus.
  4. Apply the CCA-WR deconfound transform to both the center vectors
     and the candidates, compute a deconfounded centroid, and re-rank.
     This step emphasizes responsibilities over company/style signal.
  5. Group the top-N non-center positions by normalized title and keep
     groups that meet MIN_GROUP_SIZE (so the legend isn't cluttered by
     one-offs).
  6. Lay out the final point cloud with PCA(50) -> t-SNE(2D).

Returns a dict suitable for HTML rendering (see `html_report`).

This module does not write to the database. It only reads.
"""

from __future__ import annotations

import logging
import random
import re
from collections import Counter

import numpy as np
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from . import db
from .deconfound import apply_transform, load_transform
from .title_normalize import normalize_title

log = logging.getLogger("jobproc.neighborhood")


MIN_GROUP_SIZE = 6   # a title group must have this many hits to appear
MAX_GROUPS = 12      # legend cap
MAX_CENTER_DOTS = 200  # sampled for the layout to keep t-SNE tractable


def _sql_patterns(center_patterns: list[str]) -> str:
    """Build a lax ILIKE clause that catches plural variants ("event" -> "events")."""
    clauses: set[str] = set()
    for p in center_patterns:
        clauses.add(f"lower(jl.title) LIKE '%%{p}%%'")
        words = p.split()
        for i in range(len(words)):
            variant = words[:i] + [words[i] + "s"] + words[i + 1:]
            clauses.add(f"lower(jl.title) LIKE '%%{' '.join(variant)}%%'")
    return " OR ".join(clauses)


def _fetch_center(cur, center_patterns: list[str]) -> list[dict]:
    like_clause = _sql_patterns(center_patterns)
    cur.execute(f"""
        SELECT jl.url_hash, jl.title, jl.company, jl.seniority, jl.url,
               pv.role_vec
        FROM job_listings jl
        JOIN position_vectors pv ON jl.url_hash = pv.url_hash
        WHERE jl.is_active AND jl.title != '' AND ({like_clause})
    """)
    raw = cur.fetchall()
    # ILIKE is a loose pre-filter; require the normalized title to contain
    # at least one of the center patterns so we don't drift ("product
    # manager" shouldn't also match "product marketing manager").
    center: list[dict] = []
    for row in raw:
        row = dict(row)
        row["_group"] = normalize_title(row["title"])
        g = (row["_group"] or "").lower()
        if any(p in g for p in center_patterns):
            center.append(row)
    return center


def _fetch_candidates(cur, raw_centroid: np.ndarray, limit: int) -> list[dict]:
    # Force a sequential scan for candidate selection: we often need >1000
    # candidates, which exceeds pgvector's ef_search ceiling for HNSW.
    # SET SESSION (not LOCAL) because the connection runs in autocommit
    # mode — LOCAL only binds within an explicit transaction. We RESET
    # below to leave the session clean.
    cur.execute("SET enable_indexscan = off")
    try:
        cur.execute("""
            SELECT jl.url_hash, jl.title, jl.company, jl.seniority, jl.url,
                   pv.role_vec
            FROM job_listings jl
            JOIN position_vectors pv ON jl.url_hash = pv.url_hash
            WHERE jl.is_active AND jl.title != ''
            ORDER BY pv.role_vec <=> %s::vector
            LIMIT %s
        """, ([float(x) for x in raw_centroid], limit))
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.execute("RESET enable_indexscan")


def _fetch_farthest(cur, raw_centroid: np.ndarray, n: int = 5) -> list[dict]:
    """Query pgvector for the most distant clean-English listings.

    Useful as a sanity check — the outliers should look *nothing* like
    the center role. If they look similar, something is wrong with the
    embedding space.
    """
    cur.execute("SET enable_indexscan = off")
    try:
        cur.execute("""
            SELECT jl.title, jl.company,
                   pv.role_vec <=> %s::vector AS dist
            FROM job_listings jl
            JOIN position_vectors pv ON jl.url_hash = pv.url_hash
            WHERE jl.is_active AND jl.title != ''
              AND jl.title ~ '^[A-Za-z0-9 ,\\-\\(\\)\\.&/''"]+$'
              AND length(jl.title) BETWEEN 12 AND 80
              AND jl.company IS NOT NULL AND jl.company != '' AND jl.company != 'unknown'
            ORDER BY pv.role_vec <=> %s::vector DESC
            LIMIT 200
        """, ([float(x) for x in raw_centroid], [float(x) for x in raw_centroid]))
        rows = cur.fetchall()
    finally:
        cur.execute("RESET enable_indexscan")

    english_hint = {
        "manager", "engineer", "analyst", "coordinator", "specialist",
        "associate", "assistant", "director", "lead", "head", "officer",
        "designer", "developer", "scientist", "researcher", "consultant",
        "representative", "technician", "operations", "sales", "marketing",
        "product", "data", "software", "project", "program", "senior",
        "staff", "principal",
    }
    results: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        t = r["title"]
        tl = t.lower().strip()
        words = set(re.findall(r"[a-z]+", tl))
        if len(words & english_hint) < 1:
            continue
        if tl in seen:
            continue
        seen.add(tl)
        sim = 1.0 - float(r["dist"])  # cosine
        results.append({"title": t, "company": r["company"], "sim": round(sim, 3)})
        if len(results) >= n:
            break
    return results


def find_neighborhood(
    center_patterns: list[str],
    center_label: str,
    n_neighbors: int = 700,
    random_seed: int = 42,
) -> dict | None:
    """Compute the full neighborhood visualization for one center role.

    Returns the payload consumed by `html_report.render_html`, or None
    if no center positions were found (e.g. typo in the patterns).
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    random.seed(random_seed)

    W = load_transform()
    if W is None:
        log.warning("No deconfound_transform.npz found — using raw embeddings.")

    conn = db.get_connection()
    register_vector(conn)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    center = _fetch_center(cur, center_patterns)
    log.info("Center: %d positions", len(center))
    if not center:
        return None

    # Raw centroid -> pgvector prefetch. Wide because deconfound re-ranks heavily.
    raw_center = np.stack([np.asarray(p["role_vec"], dtype=np.float32) for p in center])
    raw_centroid = raw_center.mean(axis=0)
    raw_centroid /= np.linalg.norm(raw_centroid) + 1e-10

    prefetch = max(n_neighbors * 5, n_neighbors + len(center) + 2000)
    candidates = _fetch_candidates(cur, raw_centroid, prefetch)
    log.info("Prefetched %d candidates via pgvector", len(candidates))

    center_hashes = {p["url_hash"] for p in center}
    other: list[dict] = []
    for p in candidates:
        if p["url_hash"] in center_hashes:
            continue
        p["_group"] = normalize_title(p["title"])
        if p["_group"]:
            other.append(p)

    # Deconfound both sides, then rank by cosine to the deconfounded centroid.
    center_vecs = apply_transform(raw_center, W)
    for i, p in enumerate(center):
        p["role_vec"] = center_vecs[i]
    centroid = center_vecs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-10

    other_vecs = apply_transform(
        np.stack([np.asarray(p["role_vec"], dtype=np.float32) for p in other]), W,
    )
    for i, p in enumerate(other):
        p["role_vec"] = other_vecs[i]
    sims = other_vecs @ centroid

    top_idx = np.argsort(-sims)[:n_neighbors]
    neighbors = [other[i] for i in top_idx]
    neighbor_sims = sims[top_idx]
    log.info(
        "Neighbors: %d closest (sim range %.4f - %.4f)",
        n_neighbors, float(neighbor_sims[-1]), float(neighbor_sims[0]),
    )

    title_counts = Counter(p["_group"] for p in neighbors)
    top_groups = [(t, n) for t, n in title_counts.most_common()
                  if n >= MIN_GROUP_SIZE][:MAX_GROUPS]
    top_names = {t for t, _ in top_groups}

    for t, n in top_groups:
        group_sims = [float(neighbor_sims[i]) for i, p in enumerate(neighbors)
                      if p["_group"] == t]
        log.info("  %4d  %-50s sim=%.4f", n, t, float(np.mean(group_sims)))

    # Down-sample center for the layout — 200+ dots in one color is illegible.
    center_for_layout = (
        random.sample(center, MAX_CENTER_DOTS) if len(center) > MAX_CENTER_DOTS else center
    )
    neighbor_layout = [p for p in neighbors if p["_group"] in top_names]

    for p in center_for_layout:
        p["_display"] = center_label
    for p in neighbor_layout:
        p["_display"] = p["_group"]

    viz = center_for_layout + neighbor_layout
    vecs = np.stack([np.asarray(p["role_vec"], dtype=np.float32) for p in viz])
    n = len(vecs)
    perp = min(150, max(20, n // 4))
    log.info("PCA(768->50) -> t-SNE(perplexity=%d) on %d positions", perp, n)

    pca_vecs = PCA(n_components=min(50, n - 1), random_state=random_seed).fit_transform(vecs)
    coords = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate="auto",
        init="pca",
        random_state=random_seed,
        max_iter=2000,
    ).fit_transform(pca_vecs)

    # Normalize the layout to [-1, 1] on each axis.
    for ax in (0, 1):
        mn, mx = coords[:, ax].min(), coords[:, ax].max()
        if mx - mn > 1e-9:
            coords[:, ax] = 2 * (coords[:, ax] - mn) / (mx - mn) - 1

    # "Other similar jobs" — top 20 titles that didn't make a named group.
    skip = {
        "unknown", "search", "career exploration", "job descriptions",
        "current job openings", "pro jobs", "jobs san francisco",
    }
    also_near: list[dict] = []
    seen_titles = set(top_names)
    for i in np.argsort(-neighbor_sims):
        p = other[top_idx[i]]
        t = (p.get("_group") or "").lower()
        title = p.get("title", "")
        if not t or t in seen_titles or t in skip or "jobs in" in t:
            continue
        if not title.isascii():
            continue
        if not p.get("company") or p["company"] in ("", "unknown"):
            continue
        seen_titles.add(t)
        also_near.append({
            "title": title,
            "company": p["company"],
            "sim": round(float(neighbor_sims[i]), 4),
        })
        if len(also_near) >= 20:
            break

    farthest = _fetch_farthest(cur, raw_centroid)

    group_counts = {center_label: len(center_for_layout)}
    for t, n_in_group in top_groups:
        group_counts[t] = n_in_group

    points = []
    for i, p in enumerate(viz):
        points.append({
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "title": p["title"],
            "company": p.get("company", "") or "",
            "group": p["_display"],
            "url": p.get("url", "") or "",
        })

    cur.close()
    return {
        "label": center_label,
        "description": (
            f"{n_neighbors} nearest positions to {center_label}, grouped by title"
        ),
        "points": points,
        "group_counts": group_counts,
        "farthest": farthest,
        "also_near": also_near,
    }
