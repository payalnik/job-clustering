"""Canonicalize free-form job titles into stable group labels.

A listing's raw title is noisy: seniority prefixes, trailing "@ Company"
suffixes, locations, contract types, and arbitrary capitalization.
Neighborhood analysis groups positions by normalized title, so any
cleanup we can do here improves the signal.

`normalize_title(raw) -> str | None`

Returns `None` when the title is clearly not an individual role:
  * director / VP / C-suite (too senior to cluster)
  * combo roles ("Designer / Developer" — ambiguous)
  * internships, recruiter placeholders, "dream job" marketing
  * anything under 3 characters

Otherwise returns a title cased string with acronyms preserved
("Senior AI Engineer" -> "AI Engineer", "lead ux designer" -> "UX Designer").
"""

from __future__ import annotations

import re

_SENIORITY_PREFIX = re.compile(
    r"^(?:senior|staff|principal|lead|junior|sr\.?|founding|entry[- ]level)\s+",
    re.IGNORECASE,
)
_AT_SUFFIX = re.compile(r"\s*@\s*.+$")
_CONTRACT_SUFFIX = re.compile(
    r"\s*-\s*(?:remote|contract|hybrid|onsite|on-site)$", re.IGNORECASE,
)
_CITY_SUFFIX = re.compile(r"\s*-\s*[A-Z][a-z]+(?:,?\s+[A-Z]{2})?\s*$")
_PAREN_SUFFIX = re.compile(
    r"\s*\((?:remote|contract|hybrid|m/[wfd]/d)\)\s*$", re.IGNORECASE,
)
_LEVEL_SUFFIX = re.compile(r"\s+(?:I{1,3}|IV|V|VI{0,3}|[Ll]l)$")

_DROPLIST = (
    "intern", "recrui", "cto", "ceo", "cfo",
    "work from home", "remote opportunity",
    "dream", "part time", "part-time",
)

_ACRONYMS = {
    "ai", "ml", "ui", "ux", "qa", "hr", "it", "bi", "vp", "pm",
    "sre", "api", "sdk", "etl", "crm", "erp", "seo", "sem",
    "emea", "apac", "latam", "mena", "devops", "devsecops",
    "saas", "paas", "aws", "gcp",
}


def _smart_cap(word: str) -> str:
    """Title-case a word but keep common acronyms uppercase."""
    prefix = suffix = ""
    while word and not word[0].isalnum():
        prefix += word[0]
        word = word[1:]
    while word and not word[-1].isalnum():
        suffix = word[-1] + suffix
        word = word[:-1]
    if not word:
        return prefix + suffix
    core = word.upper() if word in _ACRONYMS else word.capitalize()
    return prefix + core + suffix


def normalize_title(title: str) -> str | None:
    if not title:
        return None
    t = title.strip()
    tl = t.lower()

    # Ambiguous combo roles ("X / Y" or "X & Y") — can't place in one group.
    if "/" in tl or "&" in tl:
        return None
    if any(x in tl for x in _DROPLIST):
        return None

    t = _SENIORITY_PREFIX.sub("", t).strip()
    t = _AT_SUFFIX.sub("", t)
    t = _CONTRACT_SUFFIX.sub("", t)
    t = _CITY_SUFFIX.sub("", t)
    t = _PAREN_SUFFIX.sub("", t)
    t = _LEVEL_SUFFIX.sub("", t)

    tl = t.lower().strip()
    if tl.startswith("director") or tl.startswith("vp ") or tl.startswith("vice president"):
        return None
    if "executive director" in tl or "managing director" in tl:
        return None
    if t.count(",") > 1:
        return None

    # "events manager" / "events coordinator" -> singular
    tl = tl.replace("events ", "event ")
    if not tl or len(tl) < 3:
        return None

    return " ".join(_smart_cap(w) for w in tl.split())
