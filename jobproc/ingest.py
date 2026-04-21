"""Ingestion orchestrator: scrape every target company, upsert, expire.

Runs one full pass over `data/target_companies.json` and leaves the
database in a consistent state:

  * Listings that a company's ATS still serves are inserted (if new) or
    have their `last_refreshed` bumped.
  * Listings that a company used to serve but no longer does are marked
    `is_active = FALSE`.
  * Companies that return 404 are skipped; their existing positions are
    deactivated in bulk because the slug is clearly dead.
  * Companies that time out are skipped entirely — their positions are
    left untouched until the next run.
  * Companies whose Workday board hit the 2000-position pagination cap
    get a 24h grace period before their un-refreshed rows are expired.

Intended cadence: every 6-12h. One pass takes ~3-5 minutes on a modest
machine for the ~2,000 companies in `target_companies.json`.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import requests

from . import ats_scrapers, db
from .ats_scrapers import partial_workday_slugs
from .config import PARALLEL
from .filters import (
    clean_description, clean_location, is_aggregator_listing,
    is_garbage_description, is_garbage_url,
)

log = logging.getLogger("jobproc.ingest")


class _StaleSlug(Exception):
    """The ATS returned 404 — slug is dead."""


class _Timeout(Exception):
    """The ATS timed out after retry — skip without expiring."""


def _scrape_one(co: dict) -> tuple[str, list[dict], bool]:
    """Scrape one company. Returns (name, jobs, partial_flag).

    partial_flag is True when Workday pagination hit the 2000 cap —
    caller uses a longer grace period before expiring un-refreshed rows.
    """
    name = co["name"]
    try:
        jobs = ats_scrapers.scrape_company(co)
    except requests.exceptions.Timeout:
        try:
            log.info("RETRY  %s (timeout)", name)
            jobs = ats_scrapers.scrape_company(co)
        except requests.exceptions.Timeout:
            raise _Timeout(name)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise _StaleSlug(name)
            raise
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            raise _StaleSlug(name)
        raise
    partial = co.get("workday_slug") in partial_workday_slugs
    for j in jobs:
        j["company"] = name
        j["source_domain"] = "ats"
    return name, jobs, partial


def _sanitize(jobs: list[dict]) -> list[dict]:
    """Drop garbage, clean HTML, normalize locations."""
    out = []
    for j in jobs:
        url = j.get("url", "")
        if not url or is_garbage_url(url):
            continue
        title = j.get("title", "") or ""
        if is_aggregator_listing(title):
            continue
        desc_raw = j.get("description_raw", "") or ""
        if is_garbage_description(desc_raw):
            continue
        out.append({
            "url": url,
            "title": title,
            "company": j.get("company", ""),
            "description_raw": clean_description(desc_raw),
            "location": clean_location(j.get("location", "") or ""),
            "remote_onsite": j.get("remote_onsite", "") or "",
            "posted_date": j.get("posted_date", "") or "",
            "source_domain": "ats",
        })
    return out


def run_ingest(parallel: int = PARALLEL) -> dict:
    """Run one full ingestion pass.

    Returns a stats dict (scraped, stored, new, expired, errors, timing).
    """
    start = time.time()
    companies = ats_scrapers.load_target_companies()
    if not companies:
        log.error("No target companies found. Place target_companies.json in data/.")
        return {}
    log.info("Ingest start: %d companies, %d parallel", len(companies), parallel)

    scraped_ok: set[str] = set()
    partial_companies: set[str] = set()
    stale_slugs: list[str] = []
    timed_out: list[str] = []
    empty_companies: list[str] = []
    total_scraped = total_stored = total_new = errors = 0

    run_start = datetime.now(timezone.utc)
    run_start_iso = run_start.isoformat()

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(_scrape_one, co): co for co in companies}
        for fut in as_completed(futures):
            co = futures[fut]
            try:
                name, jobs, partial = fut.result()
            except _StaleSlug:
                stale_slugs.append(co["name"])
                log.warning("404  %s: deactivating positions", co["name"])
                db.deactivate_company(co["name"])
                continue
            except _Timeout:
                timed_out.append(co["name"])
                log.warning("TIMEOUT %s: skipping (no expiration)", co["name"])
                continue
            except Exception as e:
                errors += 1
                log.warning("ERR  %s: %s", co["name"], e)
                continue

            if not jobs:
                empty_companies.append(name)
                continue

            total_scraped += len(jobs)
            scraped_ok.add(name)
            if partial:
                partial_companies.add(name)

            clean = _sanitize(jobs)
            if clean:
                try:
                    new = db.upsert_listings(clean)
                    total_new += new
                    total_stored += len(clean)
                    if new:
                        log.info("STORE %-25s %3d new / %4d total", name, new, len(clean))
                except Exception as e:
                    errors += 1
                    log.error("DB ERR %s: %s", name, e)

            # Bump last_refreshed on every URL the ATS served (even ones we
            # filtered out) — they're still "live" as far as the company is
            # concerned. Prevents spurious expiration on the next pass.
            live_urls = [j["url"] for j in jobs if j.get("url")]
            if live_urls:
                try:
                    db.touch_refreshed(live_urls, run_start_iso)
                except Exception as e:
                    log.error("REFRESH ERR %s: %s", name, e)

    scrape_elapsed = time.time() - start
    log.info("SCRAPE %d scraped, %d stored, %d new, %d errors, %d 404s, %d timeouts (%.0fs)",
             total_scraped, total_stored, total_new, errors,
             len(stale_slugs), len(timed_out), scrape_elapsed)

    # --- Expire un-refreshed positions ---
    # Normal companies: anything not refreshed this run is gone.
    # Partial (Workday 2000 cap) companies: 24h grace. Several passes must
    # all fail to refresh a URL before we believe it's dead.
    partial_cutoff = (run_start - timedelta(hours=24)).isoformat()
    normal_cos = scraped_ok - partial_companies
    normal_expired = db.expire_unrefreshed(normal_cos, run_start_iso) if normal_cos else 0
    partial_expired = db.expire_unrefreshed(partial_companies, partial_cutoff) if partial_companies else 0
    log.info("EXPIRE %d normal + %d partial (from %d companies)",
             normal_expired, partial_expired, len(scraped_ok))

    return {
        "companies_total": len(companies),
        "scraped_ok": len(scraped_ok),
        "partial": len(partial_companies),
        "stale_404": len(stale_slugs),
        "timed_out": len(timed_out),
        "empty": len(empty_companies),
        "listings_scraped": total_scraped,
        "listings_stored": total_stored,
        "listings_new": total_new,
        "expired_normal": normal_expired,
        "expired_partial": partial_expired,
        "errors": errors,
        "elapsed_seconds": round(time.time() - start, 1),
    }
