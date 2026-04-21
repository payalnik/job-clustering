"""CCA Wiener Ridge (CCA-WR) deconfounding transform.

The raw 768-dim embeddings from `gemini-embedding-2-preview` encode
everything a job posting says: role, company voice, industry jargon,
seniority, location. That's fine for general retrieval but noisy when
what you want is *what this person would do on the job*.

The CCA-WR transform `W` is a 768x768 matrix that projects a raw
embedding into a "responsibility-focused" subspace. It was fit on
3,580 paired (role_vec, resp_vec) examples where resp_vec is a second
embedding of the same posting restricted to the responsibilities
section.

Application is a single matrix multiply:
    x_clean = x @ W
    x_clean /= ||x_clean||

On the held-out company-grouped test set this gives:
  * +22% responsibility-neighborhood preservation @20
  * +12% company invariance (same-title cross-company similarity)
  *  R^2 = 0.64 on resp_vec reconstruction

Full details: docs/METHODOLOGY.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import DECONFOUND_TRANSFORM_PATH


def load_transform(path: Path | None = None) -> np.ndarray | None:
    """Load W (768x768) from the shipped .npz. Returns None if the file is missing."""
    path = path or DECONFOUND_TRANSFORM_PATH
    if not path.exists():
        return None
    data = np.load(path)
    if "W" not in data:
        return None
    return np.asarray(data["W"], dtype=np.float32)


def apply_transform(vecs: np.ndarray, W: np.ndarray | None) -> np.ndarray:
    """Project embeddings through W (if provided) and L2-renormalize."""
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs[None, :]
    if W is not None:
        vecs = vecs @ W
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / (norms + 1e-10)).astype(np.float32)
