"""Deterministic 63-bit integer IDs for job-posting URLs.

BLAKE2b truncated to 8 bytes then masked to 63 bits so the result fits
in a PostgreSQL BIGINT. Collision probability at 300K URLs is ~2e-8 —
effectively zero for our scale.
"""

import hashlib


def url_hash(url: str) -> int:
    digest = hashlib.blake2b(url.encode(), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") & 0x7FFFFFFFFFFFFFFF
