-- PostgreSQL schema for job-processing.
-- Run once against an empty database after installing the pgvector extension.
--
--   CREATE DATABASE jobproc;
--   \c jobproc
--   CREATE EXTENSION IF NOT EXISTS vector;
--   \i schema.sql
--
-- Or use the helper: `python -m jobproc.scripts.init_db`.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per unique posting. url is the primary key; url_hash is a
-- deterministic 63-bit integer used throughout the pipeline for joins
-- and vector lookups (see jobproc/hashing.py).
CREATE TABLE IF NOT EXISTS job_listings (
    url              TEXT        PRIMARY KEY,
    url_hash         BIGINT      NOT NULL UNIQUE,
    title            TEXT        NOT NULL DEFAULT '',
    company          TEXT        NOT NULL DEFAULT '',
    description_raw  TEXT        NOT NULL DEFAULT '',
    location         TEXT        NOT NULL DEFAULT '',
    remote_onsite    TEXT        NOT NULL DEFAULT '',
    seniority        TEXT        NOT NULL DEFAULT '',
    posted_date      TEXT        NOT NULL DEFAULT '',
    source_domain    TEXT        NOT NULL DEFAULT '',
    scraped_at       TIMESTAMPTZ NOT NULL,
    last_refreshed   TIMESTAMPTZ NOT NULL,
    is_active        BOOLEAN     NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_listings_active   ON job_listings (is_active);
CREATE INDEX IF NOT EXISTS idx_listings_company  ON job_listings (company) WHERE is_active;
CREATE INDEX IF NOT EXISTS idx_listings_url_hash ON job_listings (url_hash);

-- One 768-dim vector per listing. Populated by the embedding step.
CREATE TABLE IF NOT EXISTS position_vectors (
    url_hash BIGINT       PRIMARY KEY REFERENCES job_listings(url_hash),
    role_vec VECTOR(768)
);

-- HNSW index for fast approximate cosine search. pgvector picks this
-- up automatically via `ORDER BY role_vec <=> query::vector`.
CREATE INDEX IF NOT EXISTS idx_pv_hnsw
    ON position_vectors USING hnsw (role_vec vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
