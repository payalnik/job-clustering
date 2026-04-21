#!/usr/bin/env python3
"""One full ATS ingestion pass.

Scrapes every company in data/target_companies.json via their public
Greenhouse/Lever/Ashby/Workday API, upserts listings into the database,
and marks no-longer-served URLs inactive.

Typical runtime: 3-5 minutes for the shipped ~2,000-company list.

Usage:
    python -m scripts.run_ingest
    python -m scripts.run_ingest --parallel 6
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobproc.config import PARALLEL, setup_logging
from jobproc.ingest import run_ingest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--parallel", type=int, default=PARALLEL,
        help=f"Parallel ATS requests (default: {PARALLEL})",
    )
    args = parser.parse_args()

    setup_logging()
    stats = run_ingest(parallel=args.parallel)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
