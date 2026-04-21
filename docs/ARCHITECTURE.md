# Architecture

## Data flow

```
                    ┌──────────────────────────────┐
  data/              │  data/target_companies.json │
  target_companies   └──────────────┬───────────────┘
                                     │
             ┌───────────────────────▼────────────────────────┐
             │  jobproc.ats_scrapers (Greenhouse/Lever/       │
             │  Ashby/Workday public APIs, parallel I/O)      │
             └───────────────────────┬────────────────────────┘
                                     │  list[dict] per company
             ┌───────────────────────▼────────────────────────┐
             │  jobproc.ingest (filter + upsert + expire)     │
             └───────────────────────┬────────────────────────┘
                                     │
                   ┌──── PostgreSQL ─┴──── pgvector ────┐
                   │  job_listings       position_vectors│
                   │  (url, title, …)    (url_hash,      │
                   │                      role_vec[768]) │
                   └──── ▲ ──────────────── ▲ ───────────┘
                         │                  │
                         │     ┌────────────┴──────────────┐
                         │     │  jobproc.encode (Gemini   │
                         │     │  embed_content, batches   │
                         │     │  of 100, 3s throttle)     │
                         │     └───────────────────────────┘
                         │
             ┌───────────┴─────────────────────────────────────┐
             │  jobproc.neighborhood                            │
             │    fetch center by SQL ILIKE                     │
             │    pgvector seq-scan for N closest by raw cosine │
             │    deconfound (x @ W, L2 renorm)                 │
             │    re-rank by deconfounded cosine                │
             │    group by normalize_title()                    │
             │    PCA(50) -> t-SNE(2D)                          │
             └───────────────────────┬─────────────────────────┘
                                     │
             ┌───────────────────────▼────────────────────────┐
             │  jobproc.html_report → out/.../index.html      │
             └────────────────────────────────────────────────┘
```

## Database

Two tables, no views, no triggers. `schema.sql` is the source of truth.

### `job_listings`

One row per posting. `url` is the primary key; `url_hash` is the
deterministic 63-bit ID used for joins and referenced by the embeddings
table.

| column           | type          | purpose                                      |
|------------------|---------------|----------------------------------------------|
| url              | text PK       | canonical URL (tracking params stripped)     |
| url_hash         | bigint UNIQUE | blake2b truncated to 63 bits                 |
| title            | text          | raw title from the ATS                       |
| company          | text          | company name                                 |
| description_raw  | text          | HTML-stripped description                    |
| location         | text          | free-form location string                    |
| remote_onsite    | text          | "remote" / "hybrid" / "onsite" / ""          |
| seniority        | text          | reserved; not populated by the current pipeline |
| posted_date      | text          | ISO-ish, best effort                         |
| source_domain    | text          | always "ats" for this pipeline               |
| scraped_at       | timestamptz   | first-seen timestamp                         |
| last_refreshed   | timestamptz   | most-recent re-scrape                        |
| is_active        | bool          | false when the ATS stopped serving this URL  |

### `position_vectors`

One row per embedded listing.

| column   | type         | purpose                              |
|----------|--------------|--------------------------------------|
| url_hash | bigint PK FK | references job_listings.url_hash     |
| role_vec | vector(768)  | Gemini embedding, L2-normalized      |

An HNSW index on `role_vec` with cosine ops makes top-K retrieval
milliseconds for small K. For the neighborhood prefetch (N=3000+) we
bypass the index with `SET enable_indexscan = off` — HNSW caps at a few
dozen results regardless of LIMIT.

## Connection strategy

`jobproc/db.py` keeps **one psycopg2 connection per thread**, opened
lazily on first use, with `autocommit=True`. There is no pool.

Why:
* `autocommit=True` prevents "idle in transaction" states that block
  VACUUM and leak connections when a worker dies.
* Thread-local connections are simpler than pgbouncer at this scale
  (PARALLEL=10 threads × a handful of server workers).
* Every `cursor()` uses `RealDictCursor`, so downstream code deals in
  plain dicts and never cares about column order.

## Ingestion semantics

`jobproc.ingest.run_ingest` does one pass:

1. Load `target_companies.json`.
2. Fan out to `ats_scrapers.scrape_company` in a `ThreadPoolExecutor`.
3. For each successful company, sanitize + upsert listings. Every URL
   the ATS served (including ones we filtered out) has its
   `last_refreshed` bumped — they're still "live".
4. At the end, any row in a scraped company whose `last_refreshed` is
   older than this run's start time is marked inactive.
5. Workday companies that hit the 2000-position pagination cap use a
   24-hour grace period instead — several consecutive runs must fail
   to refresh a URL before we believe it's dead.
6. 404 responses mark all of that company's rows inactive immediately.
7. Timeouts are skipped — the company's state is untouched.

This gives correct semantics under partial failures: the next pass
picks up exactly where the last one left off, and no single flaky
company can wipe out the whole corpus.

## Deconfounding in one paragraph

`data/deconfound_transform.npz` contains a 768×768 matrix `W` trained
with canonical correlation analysis + Wiener ridge regression on 3,580
paired (role_vec, resp_vec) examples, where `resp_vec` embeds only the
responsibilities section of the same posting. Applying `x @ W` and L2
renormalizing projects a raw embedding into a subspace where company
voice, industry jargon, and title wording are suppressed and
responsibilities dominate. On held-out company-grouped test folds this
gives +22% responsibility-neighborhood preservation @20, +12% company
invariance, and R²=0.64 on `resp_vec` reconstruction. Full derivation
in [METHODOLOGY.md](METHODOLOGY.md).

## Neighborhood algorithm

For each center role (e.g. "Data Scientist"):

1. **Find center positions** — SQL `ILIKE` against raw titles, then a
   second pass in Python using `normalize_title` to enforce the exact
   pattern. SQL LIKE is a loose pre-filter; the normalized-title check
   prevents drift ("product marketing manager" matching "product manager").
2. **Raw centroid** — mean of center vectors, L2-normalized. Used only
   as a probe into pgvector; not returned to the caller.
3. **Prefetch** — `ORDER BY role_vec <=> centroid LIMIT N` with
   `enable_indexscan = off`. Returns N=3,000+ candidates from the full
   active corpus via seq scan (~few hundred ms on 90K rows).
4. **Deconfound both sides** — apply `W` to center + candidate vectors,
   L2-renorm. Compute a deconfounded centroid from the center vectors.
5. **Re-rank** by cosine to the deconfounded centroid; keep top N.
6. **Group** by `normalize_title`. Groups with ≥6 members and ≤12 total
   get their own colored hull in the visualization.
7. **Lay out** with PCA(768→50) → t-SNE(50→2D). `random_state=42`
   everywhere so the same query gives the same layout.

## Extension points

* **Swap the embedding model** — edit `_MODEL` and `_DIMS` in
  `jobproc/embedder.py` and retrain the deconfound transform on the
  new space. The schema's `vector(768)` type will need to change if
  you pick a different dimensionality.
* **Add location filtering** — put the filter in the SQL WHERE clause
  of `_fetch_candidates`, *before* the `ORDER BY`. Never post-filter
  the HNSW top-K — it starves rare locations.
* **Daily refresh** — wire `scripts/run_ingest.py` into cron every 6-12h
  and `scripts/run_encode.py` after it. The ingest is idempotent and
  the encoder skips already-embedded rows, so partial failures are fine.
