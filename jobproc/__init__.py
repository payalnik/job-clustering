"""Job-listing ingestion and embedding neighborhood analysis.

Public modules:
    jobproc.config           configuration (DB DSN, API keys, paths)
    jobproc.db               minimal PostgreSQL wrapper
    jobproc.ats_scrapers     Greenhouse / Lever / Ashby / Workday scrapers
    jobproc.ingest           orchestrator: scrape -> upsert -> expire
    jobproc.embedder         Gemini embedding API wrapper
    jobproc.encode           batch encode un-embedded positions
    jobproc.deconfound       CCA Wiener Ridge transform loader
    jobproc.title_normalize  canonical job-title normalizer
    jobproc.neighborhood     nearest-neighbor discovery + visualization
"""

__version__ = "0.1.0"
