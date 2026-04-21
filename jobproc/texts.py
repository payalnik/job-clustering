"""Build the text that gets handed to the embedding model.

Only the `role_text` function is used by this repo. The goal is to feed
the embedder the *role-relevant* portion of each job listing, not the
boilerplate "About us" paragraphs that typically precede it.

Heuristic:
  1. If the description has a clear role-section header ("Responsibilities",
     "What you'll do", etc.) in its first half, snip from there.
  2. Otherwise, take the first 500 chars (for context) plus another window
     from the middle where responsibilities typically live.
  3. Prepend the title so short/generic descriptions still get a signal.

The 2000-char cap matches what the deconfound transform was trained on.
"""

from __future__ import annotations

import re

_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

_ROLE_START = re.compile(
    r"(?:about the (?:role|position|job|opportunity)|"
    r"what you'?ll do|what you'?ll be doing|"
    r"the role|role overview|role summary|"
    r"job description|position summary|position overview|"
    r"key responsibilities|primary responsibilities|core responsibilities|"
    r"responsibilities|about this role|about this position|"
    r"in this role|your role|the opportunity|"
    r"what we'?re looking for|we are looking for|we'?re hiring|"
    r"(?:key |primary )?duties)",
    re.IGNORECASE,
)


def _strip_html(text: str) -> str:
    return _WS.sub(" ", _HTML_TAG.sub(" ", text)).strip()


def extract_role_text(description: str, max_chars: int = 2000) -> str:
    """Return the most role-relevant slice of a description, max_chars long."""
    clean = _strip_html(description)
    if len(clean) <= max_chars:
        return clean

    m = _ROLE_START.search(clean)
    if m and m.start() < len(clean) // 2:
        return clean[m.start():m.start() + max_chars]

    if len(clean) > 3000:
        return clean[:500] + " " + clean[1000:1000 + max_chars - 500]
    return clean[:max_chars]


def role_text(title: str, description: str) -> str:
    """Combine title + best-effort description slice into one embedding input."""
    title = (title or "").strip()
    desc = (description or "").strip()
    if not desc:
        return title
    return f"{title}. {extract_role_text(desc)}"
