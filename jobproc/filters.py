"""Content-quality filters for job listings.

These run at ingestion time to reject aggregator landing pages, search
result URLs, and navigation boilerplate that would otherwise pollute
the embedding index. Each function is pure — no DB or network access.

Derived from the production filters used in the Next Play crawler;
see the upstream discussion in docs/METHODOLOGY.md.
"""

from __future__ import annotations

import re
import urllib.parse


# --- Titles ------------------------------------------------------------------

_AGGREGATOR_NAMES = (
    r"LinkedIn|Indeed|Glassdoor|ZipRecruiter|Wellfound|"
    r"Craigslist|Zippia|Career\.io|CareerBuilder|Dice"
)


def is_aggregator_listing(title: str) -> bool:
    """True if the title is an index/search/landing page, not a real posting."""
    t = (title or "").strip()
    if not t or t.lower() == "unknown":
        return True
    tl = t.lower()

    if re.match(r"^\d[\d,]*\+?\s+\w+.*\bjobs?\b", tl):
        return True
    if re.match(r"^(find|browse|search|best|latest|top|new)\s+\w+.*\bjobs?\b", tl):
        return True
    if re.match(r"^[A-Z][\w\s,]{2,30}\s+hiring\s+[A-Z]", t) and "HIRING NOW" not in t.upper():
        return True
    if re.match(r"^(careers?|jobs?)\s+(home|opportunit|opening|search|at\s)", tl):
        return True
    if re.match(r"^join\s+(our|us)\b", tl) or re.match(r"^explore\s+career", tl):
        return True
    if re.match(r"^work\s+in\s+\w+\s*[-–]?\s*careers?", tl):
        return True
    if re.search(r'["\u201c].+?["\u201d].*\bjobs\b', tl):
        return True
    if re.match(r"^[\d,]+\s+\w[\w\s]*\bjobs?\s+in\s+", tl) and len(t) < 100:
        return True

    suffix = r"[\|–-]\s*(" + _AGGREGATOR_NAMES + r")\s*$"
    if re.search(suffix, t):
        prefix = re.sub(suffix, "", t).strip()
        if re.search(r"\bjobs?\b", prefix.lower()):
            return True
    if re.match(r"^(multiple|various)\s+(positions|openings|roles|opportunities)", tl):
        return True
    if re.match(r"^(search|all|open|view|see)\s+(jobs?|positions|roles|openings)", tl):
        return True
    if re.search(r"\bjobs?\s*[&+]\s*careers?\b", tl):
        return True
    if re.match(r"^[\w\s]+(careers?|jobs?)\s*[|–-]\s*(do|join|build|discover|find)", tl):
        return True
    return False


# --- URLs --------------------------------------------------------------------

def is_garbage_url(url: str) -> bool:
    """True if the URL is a search/category page, not an individual listing."""
    u = (url or "").lower()
    if re.search(r"/search\?", u):
        return True
    if re.search(r"/job-categories/", u):
        return True
    if re.search(r"/teams/", u) and not re.search(r"/jobs?/", u):
        return True
    if "jobs.lever.co/" in u:
        parts = urllib.parse.urlparse(u).path.strip("/").split("/")
        if len(parts) < 2:  # board root, not a posting
            return True
    if "leverdemo" in u:
        return True
    return False


def clean_ats_url(url: str) -> str:
    """Strip tracking params and /application suffixes from ATS URLs."""
    if not url:
        return url
    if "linkedin.com/jobs/view/" in url:
        url = re.sub(r"\?.*$", "", url)
    if "ashbyhq.com/" in url:
        url = re.sub(r"/application\b.*$", "", url)
        url = re.sub(r"\?.*$", "", url)
    if "greenhouse.io/" in url or "?gh_jid=" in url:
        url = re.sub(r"\?.*$", "", url)
    if "jobs.lever.co/" in url:
        url = re.sub(r"\?.*$", "", url)
    if "google.com/" in url and "/careers/" in url:
        url = re.sub(r"\?.*$", "", url)
    return url


# --- Descriptions ------------------------------------------------------------

def is_garbage_description(desc: str) -> bool:
    """True if the 'description' is navigation, a login wall, or JS-only."""
    if not desc or len(desc) < 100:
        return True
    d = desc.strip()
    if "My career" in d and "My applications" in d and "My profile" in d:
        return True
    if d.startswith("Sign in") or d.startswith("Log in") or d.startswith("Accept cookies"):
        return True
    js_markers = (
        "please enable javascript", "you need to enable javascript",
        "javascript is required", "this site requires javascript",
    )
    if len(d) < 500 and any(m in d.lower() for m in js_markers):
        return True
    return False


def clean_description(desc: str) -> str:
    """Strip HTML tags and entities, collapse whitespace."""
    if not desc:
        return desc
    desc = desc.replace("&nbsp;", " ").replace("&amp;", "&")
    desc = desc.replace("&lt;", "<").replace("&gt;", ">")
    desc = desc.replace("&quot;", '"').replace("&#39;", "'")
    desc = re.sub(r"&#\d+;", " ", desc)
    desc = re.sub(r"&\w+;", " ", desc)
    desc = re.sub(r"<[^>]+>", " ", desc)
    return re.sub(r"\s+", " ", desc).strip()


def clean_location(loc: str) -> str:
    """Drop obvious location garbage ('2 Locations', 'Hybrid', etc.)."""
    if not loc:
        return loc
    loc = loc.strip()
    if re.match(r"^\d+\s+Locations?$", loc, re.IGNORECASE):
        return ""
    if loc.lower() in (
        "hybrid", "onsite", "on-site", "n/a", "various",
        "multiple", "tbd", "flexible", "see description",
    ):
        return ""
    return loc
