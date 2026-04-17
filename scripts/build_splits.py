#!/usr/bin/env python
"""Build parquet split manifests from a YAML config.

Usage:
  python scripts/build_splits.py --config configs/splits/batch01_only.yaml [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from splits.builder import build_splits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing split dir")
    args = ap.parse_args()
    out = build_splits(args.config, force=args.force)
    print(f"[build_splits] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
