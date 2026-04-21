#!/usr/bin/env python3
"""Embed every active listing that doesn't yet have a vector.

Requires a live GEMINI_API_KEY (see .env.example).

At ~100 positions per batch and a 3-second throttle between batches,
50K positions takes roughly 25 minutes. Safe to stop and resume — the
script picks up where it left off on the next run.

Usage:
    python -m scripts.run_encode
    python -m scripts.run_encode --batch-size 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobproc.config import EMBED_BATCH, setup_logging
from jobproc.encode import encode_new_positions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--batch-size", type=int, default=EMBED_BATCH,
        help=f"Positions per Gemini embed call (max 100, default: {EMBED_BATCH})",
    )
    args = parser.parse_args()

    setup_logging()
    encoded = encode_new_positions(batch_size=args.batch_size)
    print(f"Encoded {encoded} positions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
