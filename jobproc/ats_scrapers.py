"""Direct ATS API scrapers: Greenhouse, Lever, Ashby, Workday.

These hit the *public* JSON endpoints that each ATS exposes for its
embedded careers widget. No HTML parsing, no headless browser, no auth.

Each scraper takes a company's ATS "slug" and returns a list of job
dicts ready for `jobproc.db.upsert_listings`:

    {
        "title": str,
        "url": str,
        "description_raw": str,
        "location": str,
        "remote_onsite": str,  # "remote" | "hybrid" | "onsite" | ""
        "posted_date": str,    # ISO-ish, best effort
    }

A 404 propagates as an `HTTPError` so the caller can mark the slug as
stale. A timeout propagates as `requests.exceptions.Timeout` so the
caller can retry. Other exceptions are swallowed and logged — one
broken company shouldn't tank a full ingest run.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .config import REQUEST_TIMEOUT, TARGET_COMPANIES_PATH, USER_AGENT
from .filters import is_aggregator_listing

log = logging.getLogger("jobproc.ats")

_WD_PAGE_SIZE = 20       # Workday hard limit; 21+ returns HTTP 400
_WD_MAX_TOTAL = 2000     # Workday silently caps offset pagination at 2000
_WD_HEADERS = {"Content-Type": "application/json", "User-Agent": USER_AGENT}

# Set of workday slugs whose last scrape hit the 2000 pagination cap.
# Populated during scraping; consumed by the expiry logic in ingest.py.
partial_workday_slugs: set[str] = set()


def load_target_companies(path: Path | None = None) -> list[dict]:
    """Load the curated list of companies + their ATS slugs."""
    path = path or TARGET_COMPANIES_PATH
    if not path.exists():
        return []
    return json.loads(path.read_text())


# --- Greenhouse --------------------------------------------------------------

def scrape_greenhouse(slug: str) -> list[dict]:
    resp = requests.get(
        f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true",
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    jobs = []
    for job in resp.json().get("jobs", []):
        content = job.get("content", "") or ""
        desc = BeautifulSoup(content, "html.parser").get_text("\n", strip=True) if content else ""
        loc = job.get("location", {}) or {}
        jobs.append({
            "title": job.get("title", ""),
            "url": job.get("absolute_url", ""),
            "description_raw": desc,
            "location": loc.get("name", "") if isinstance(loc, dict) else str(loc),
            "remote_onsite": "",
            "posted_date": job.get("first_published", "") or "",
        })
    return jobs


# --- Lever -------------------------------------------------------------------

def scrape_lever(slug: str) -> list[dict]:
    resp = requests.get(
        f"https://api.lever.co/v0/postings/{slug}",
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        return []
    jobs = []
    for posting in data:
        title = (posting.get("text") or "").strip()
        if not title or is_aggregator_listing(title):
            continue
        desc = posting.get("descriptionPlain") or posting.get("description") or ""
        if not desc:
            # Lever's "lists" are sectioned content; flatten with bullet markers.
            parts: list[str] = []
            for lst in posting.get("lists", []) or []:
                parts.append(lst.get("text", ""))
                for item in (lst.get("content") or "").split("<li>"):
                    clean = item.replace("</li>", "").strip()
                    if clean:
                        parts.append(f"- {clean}")
            desc = "\n".join(parts)
        cats = posting.get("categories") or {}
        loc = cats.get("location", "") if isinstance(cats, dict) else ""
        workplace = posting.get("workplaceType", "")
        remote = {"remote": "remote", "hybrid": "hybrid"}.get(workplace, "")
        url = posting.get("hostedUrl") or f"https://jobs.lever.co/{slug}/{posting.get('id', '')}"
        jobs.append({
            "title": title,
            "url": url,
            "description_raw": desc,
            "location": loc,
            "remote_onsite": remote,
            "posted_date": str(posting.get("createdAt", ""))[:10],
        })
    return jobs


# --- Ashby -------------------------------------------------------------------

def scrape_ashby(slug: str) -> list[dict]:
    resp = requests.get(
        f"https://api.ashbyhq.com/posting-api/job-board/{slug}",
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    jobs = []
    for job in data.get("jobs", []) or []:
        desc = job.get("descriptionPlain") or ""
        if not desc and job.get("descriptionHtml"):
            desc = BeautifulSoup(job["descriptionHtml"], "html.parser").get_text("\n", strip=True)
        url = job.get("jobUrl") or job.get("applyUrl") or ""
        if not url and job.get("id"):
            url = f"https://jobs.ashbyhq.com/{slug}/{job['id']}"
        is_remote = job.get("isRemote")
        workplace = (job.get("workplaceType") or "").lower()
        remote = "remote" if is_remote else workplace if workplace else ""
        jobs.append({
            "title": job.get("title", ""),
            "url": url,
            "description_raw": desc,
            "location": job.get("locationName") or job.get("location", ""),
            "remote_onsite": remote,
            "posted_date": str(job.get("publishedAt", ""))[:10],
        })
    return jobs


# --- Workday -----------------------------------------------------------------

def scrape_workday(slug: str, known_urls: set[str] | None = None) -> list[dict]:
    """Scrape Workday via its public "/wday/cxs" JSON endpoint.

    `slug` has the form "company/site" (e.g. "jpmorgan/jobs"). The final
    URL pattern is:
        https://{company}.wd5.myworkdayjobs.com/wday/cxs/{company}/{site}/jobs

    Workday uses 5+ subdomains in the wild (wd1..wd5, wd103, ...). We
    probe them until one returns HTTP 200, then paginate.

    Detail pages (the job description) are only fetched for URLs not in
    `known_urls`. Pass the set of already-indexed URLs to avoid hammering
    Workday on every refresh.
    """
    parts = slug.split("/", 1)
    if len(parts) != 2:
        return []
    company, site = parts

    api_base = None
    base_url = None
    for wd in ("wd5", "wd1", "wd3", "wd503", "wd12", "wd103"):
        url = f"https://{company}.{wd}.myworkdayjobs.com/wday/cxs/{company}/{site}/jobs"
        try:
            resp = requests.post(
                url,
                json={"limit": _WD_PAGE_SIZE, "offset": 0, "searchText": ""},
                headers=_WD_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                base_url = f"https://{company}.{wd}.myworkdayjobs.com"
                api_base = f"{base_url}/wday/cxs/{company}/{site}"
                break
        except requests.exceptions.RequestException:
            continue
    if api_base is None:
        raise requests.exceptions.HTTPError(
            f"404 no working Workday subdomain for {slug}",
        )

    data = resp.json()
    total = data.get("total", 0) or 0
    if total > _WD_MAX_TOTAL:
        partial_workday_slugs.add(slug)
        log.warning("Workday partial scrape %s: %d > %d cap", slug, total, _WD_MAX_TOTAL)

    postings = list(data.get("jobPostings", []) or [])
    page_limit = min(total, _WD_MAX_TOTAL)
    offset = _WD_PAGE_SIZE
    while offset < page_limit:
        try:
            page = requests.post(
                f"{api_base}/jobs",
                json={"limit": _WD_PAGE_SIZE, "offset": offset, "searchText": ""},
                headers=_WD_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            if page.status_code != 200:
                break
            new_postings = page.json().get("jobPostings", []) or []
            if not new_postings:
                break
            postings.extend(new_postings)
            offset += _WD_PAGE_SIZE
        except requests.exceptions.RequestException:
            break

    known_urls = known_urls or set()
    jobs = []
    for item in postings:
        title = item.get("title", "")
        ext_path = item.get("externalPath", "")
        job_url = f"{base_url}/{site}{ext_path}" if ext_path else ""
        loc = item.get("locationsText", "") or ""
        if not isinstance(loc, str):
            loc = ""
        desc = ""
        if ext_path and job_url not in known_urls:
            try:
                detail = requests.get(
                    f"{api_base}{ext_path}",
                    headers=_WD_HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )
                if detail.status_code == 200:
                    info = detail.json().get("jobPostingInfo", {}) or {}
                    desc = info.get("jobDescription", "") or ""
                    if desc:
                        desc = re.sub(r"<[^>]+>", " ", desc)
                        desc = re.sub(r"\s+", " ", desc).strip()
            except requests.exceptions.RequestException:
                pass
        jobs.append({
            "title": title,
            "url": job_url,
            "description_raw": desc,
            "location": loc,
            "remote_onsite": "",
            "posted_date": item.get("postedOn", "") or "",
        })
    return jobs


# --- Dispatcher --------------------------------------------------------------

def scrape_company(co: dict, known_urls: set[str] | None = None) -> list[dict]:
    """Scrape whichever ATS the company lists a slug for.

    Order of preference: Greenhouse -> Lever -> Ashby -> Workday. Returns
    an empty list for companies with no supported slug.
    """
    try:
        if co.get("greenhouse_slug"):
            return scrape_greenhouse(co["greenhouse_slug"])
        if co.get("lever_slug"):
            return scrape_lever(co["lever_slug"])
        if co.get("ashby_slug"):
            return scrape_ashby(co["ashby_slug"])
        if co.get("workday_slug"):
            return scrape_workday(co["workday_slug"], known_urls=known_urls)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            raise
        log.warning("ATS HTTP error %s: %s", co.get("name"), e)
    except requests.exceptions.Timeout:
        raise
    except Exception as e:
        log.warning("ATS unexpected error %s: %s", co.get("name"), e)
    return []
