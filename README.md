# job-processing

End-to-end pipeline for building an embedding-based **job neighborhood atlas**:

1. **Ingest** — scrape every listing from 2,000+ companies' public ATS APIs
   (Greenhouse, Lever, Ashby, Workday). One JSON write per listing, deduped
   by canonical URL, garbage-filtered at the entry point.
2. **Embed** — encode each listing with Gemini's 768-dim embedding model,
   stored in PostgreSQL via `pgvector` (HNSW index on cosine distance).
3. **Deconfound** — project raw embeddings through a CCA Wiener Ridge
   transform (`data/deconfound_transform.npz`) that emphasizes
   responsibilities over company voice and job-title wording.
4. **Discover** — for a given center role, fetch its nearest N neighbors
   in the deconfounded space, group them by normalized title, lay them
   out with t-SNE, and render a self-contained HTML report.

The neighborhood output is what drives the interactive visualization of
adjacent roles ("Data Scientist" → Data Engineer, ML Engineer, Analytics
Engineer, …).

## Quickstart

```bash
# 1. Install dependencies (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env                 # edit DB creds + GEMINI_API_KEY
createdb jobproc                     # or point JOBPROC_DB_NAME at an existing DB
psql jobproc -c 'CREATE EXTENSION vector'

# 3. Create tables
python -m scripts.init_db

# 4. Ingest listings (~3-5 minutes)
python -m scripts.run_ingest

# 5. Generate embeddings (~25 minutes for 50K new positions)
python -m scripts.run_encode

# 6. Build the neighborhood report
python -m scripts.run_neighborhood --role ds
open out/neighborhood/index.html
```

## What's in the box

```
job-processing/
  README.md                # this file
  pyproject.toml           # package metadata
  requirements.txt         # pip install -r requirements.txt
  .env.example             # copy to .env
  schema.sql               # PostgreSQL schema (run by scripts/init_db.py)
  data/
    target_companies.json  # ~2,000 companies + their ATS slugs
    deconfound_transform.npz  # pre-trained CCA-WR W matrix (768x768)
  jobproc/                 # the library
    config.py              # env-driven settings (DB DSN, API keys)
    db.py                  # thread-local PG connection + batch upserts
    hashing.py             # url_hash — deterministic 63-bit BIGINT ID
    filters.py             # garbage URL / aggregator / boilerplate rejection
    ats_scrapers.py        # Greenhouse / Lever / Ashby / Workday
    ingest.py              # full ingestion pass: scrape -> upsert -> expire
    texts.py               # role-text extraction from raw descriptions
    embedder.py            # Gemini embedding API wrapper
    encode.py              # encode un-embedded listings in batches
    deconfound.py          # load + apply the CCA-WR transform
    title_normalize.py     # canonical job-title normalizer
    neighborhood.py        # nearest-neighbor discovery for a center role
    html_report.py         # self-contained HTML visualization
  scripts/
    init_db.py             # apply schema.sql
    run_ingest.py          # scrape + upsert + expire
    run_encode.py          # embed every un-embedded active listing
    run_neighborhood.py    # build the HTML report
  tests/                   # pytest — pure-function tests, no DB/network
  docs/
    METHODOLOGY.md         # how the deconfound transform was trained
    ARCHITECTURE.md        # data flow, schema, and design decisions
```

## Requirements

* **Python 3.11+**
* **PostgreSQL 16+** with the [`pgvector`](https://github.com/pgvector/pgvector)
  extension installed at the server level.
* **Gemini API key** — get one at <https://aistudio.google.com/apikey>.
  Roughly $0.20 per 1M embedding tokens; encoding 50K listings costs
  $2-$4 depending on description length.
* ~200 MB disk for the database at 50K listings; most of that is vectors.

## How to use the library directly

The CLIs in `scripts/` are thin wrappers; the library is the real API.

```python
from jobproc import ingest, encode, neighborhood, html_report
from pathlib import Path

# Run one ingestion pass
stats = ingest.run_ingest(parallel=10)

# Encode everything not yet encoded
encode.encode_new_positions()

# Build one visualization
viz = neighborhood.find_neighborhood(
    center_patterns=["data scientist"],
    center_label="Data Scientist",
    n_neighbors=700,
)
html_report.render_html({"ds": viz}, Path("out/ds.html"))
```

## Adding your own center roles

`scripts/run_neighborhood.py --custom` adds one without touching source:

```bash
python -m scripts.run_neighborhood \
    --custom sre:"Site Reliability Engineer":"sre|site reliability engineer"
```

For permanent roles, edit `BUILTIN_ROLES` in `scripts/run_neighborhood.py`.
`center_patterns` is a list of substring matches against the normalized
title (lowercased, stripped of seniority/suffixes — see
`jobproc/title_normalize.py`).

## Adding your own companies

`data/target_companies.json` is a list of dicts:

```json
{
  "name": "Anthropic",
  "greenhouse_slug": "anthropic",
  "lever_slug": null,
  "ashby_slug": null,
  "workday_slug": null,
  "industry": "AI / machine learning",
  "stage": "growth",
  "size": "500-1000"
}
```

Only one of the four `*_slug` fields needs to be set. Look at the
company's public careers page, inspect the iframe/XHR that loads jobs,
and copy the slug from there. A 404 on the first run means the slug is
stale — the company has moved to a different ATS.

## Reproducibility notes

* **`random_state=42`** is fixed throughout (t-SNE, PCA, center sampling).
  Re-running the same query on the same database returns the same layout.
* **Gemini embeddings are deterministic up to precision** — same model +
  same text + same `output_dimensionality` returns the same vector.
  The model ID (`gemini-embedding-2-preview`) and dims (768) are locked
  in `jobproc/embedder.py`; do not change without retraining the
  deconfound transform.
* **The `.npz` transform is versioned** — the shipped file stores a 768×768
  `W` matrix under key `"W"` plus a `"version"` scalar. The loader checks
  for `"W"`; future retrained versions stay drop-in compatible.

## Design decisions worth preserving

See `docs/ARCHITECTURE.md` for the longer form. A few high-value ones:

* **Location filter stays IN the SQL query, never post-filter.** The current
  neighborhood code doesn't filter by location, but if you add one,
  filtering after HNSW top-K starves rare locations. See the upstream
  commit history in the Next Play repo for the full debugging trail.
* **Never store listings without meaningful content.** `upsert_listings`
  drops rows where `title == '' AND len(description) < 50`. Empty text
  crashes the embedder and pollutes downstream search.
* **`autocommit=True`, no connection pool.** Thread-local connections +
  autocommit keeps us from leaking "idle in transaction" sessions and
  removes the need for pgbouncer at this scale. See `jobproc/db.py`.
* **`SET enable_indexscan = off` is required** before the neighborhood
  prefetch query. HNSW caps at a few tens of results regardless of
  `LIMIT`; you need the seq scan to get 1K+ candidates. See
  `jobproc/neighborhood.py:_fetch_candidates`.

## Tests

```bash
pip install -e '.[dev]'
pytest -q
```

21 tests, all pure-function (no DB, no network). They cover
title normalization, URL/title/description filtering, and hashing.

## License

MIT.
