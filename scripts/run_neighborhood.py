#!/usr/bin/env python3
"""Compute the neighborhood visualization for one or more center roles.

Reads embeddings from the database, applies the CCA-WR deconfound
transform, finds N nearest positions to each center, groups by
normalized title, lays them out with t-SNE, and writes a single
self-contained HTML file plus a JSON dump of the same data.

Usage:
    python -m scripts.run_neighborhood                    # all built-in roles
    python -m scripts.run_neighborhood --role ds          # Data Scientist only
    python -m scripts.run_neighborhood --role pm --n 500  # custom neighbor count
    python -m scripts.run_neighborhood \\
        --custom swe:"Software Engineer":"software engineer"

`--custom` lets you add ad-hoc roles without editing the source:
    <key>:<label>:<pattern>[|<pattern>...]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobproc.config import setup_logging
from jobproc.html_report import render_html
from jobproc.neighborhood import find_neighborhood


BUILTIN_ROLES: dict[str, dict] = {
    "ds":  {"label": "Data Scientist",      "center_patterns": ["data scientist"]},
    "swe": {"label": "Software Engineer",   "center_patterns": ["software engineer"]},
    "pm":  {"label": "Product Manager",     "center_patterns": ["product manager"]},
    "ux":  {"label": "UX / User Experience",
            "center_patterns": ["ux designer", "ux researcher", "user experience designer"]},
    "em":  {"label": "Event Manager",       "center_patterns": ["event manager"]},
    "ai":  {"label": "AI Engineer",         "center_patterns": ["ai engineer"]},
    "pjm": {"label": "Project Manager",     "center_patterns": ["project manager"]},
    "tpm": {"label": "Technical Program Manager",
            "center_patterns": ["technical program manager"]},
}


def _parse_custom(spec: str) -> tuple[str, dict]:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise SystemExit(f"--custom expects 'key:label:pattern1|pattern2', got {spec!r}")
    key, label, patterns = parts
    center_patterns = [p.strip().lower() for p in patterns.split("|") if p.strip()]
    if not center_patterns:
        raise SystemExit(f"--custom has no patterns: {spec!r}")
    return key, {"label": label, "center_patterns": center_patterns}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--role", choices=list(BUILTIN_ROLES.keys()),
                        help="Run just one built-in role (default: all).")
    parser.add_argument("--n", type=int, default=700,
                        help="Number of nearest neighbors to consider per role.")
    parser.add_argument("--custom", action="append", default=[],
                        help="Add a custom role: 'key:Label:pattern1|pattern2'.")
    parser.add_argument(
        "--out", default="out/neighborhood",
        help="Directory for index.html + data.json (default: ./out/neighborhood).",
    )
    args = parser.parse_args()

    setup_logging()
    roles: dict[str, dict] = {}
    if args.role:
        roles[args.role] = BUILTIN_ROLES[args.role]
    elif not args.custom:
        roles = BUILTIN_ROLES
    for spec in args.custom:
        k, cfg = _parse_custom(spec)
        roles[k] = cfg

    vizzes: dict[str, dict] = {}
    for key, cfg in roles.items():
        print(f"\n=== {key} — {cfg['label']} ===")
        viz = find_neighborhood(
            cfg["center_patterns"], cfg["label"], n_neighbors=args.n,
        )
        if viz:
            vizzes[key] = viz

    if not vizzes:
        print("No visualizations generated.")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_html(vizzes, out_dir / "index.html")
    (out_dir / "data.json").write_text(json.dumps(vizzes, default=str))
    print(f"\nWrote {out_dir / 'index.html'}")
    print(f"      {out_dir / 'data.json'}  ({len(vizzes)} roles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
