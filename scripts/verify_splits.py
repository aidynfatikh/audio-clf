#!/usr/bin/env python
"""Sanity-check a split directory.

Verifies:
  - all three parquet files exist
  - speaker disjointness (if speaker_disjoint was set in config)
  - no (dataset,row_id) overlap across splits
  - summary.json counts match parquet row counts
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from splits.io import read_manifests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True, type=Path)
    args = ap.parse_args()
    split_dir = args.split_dir.resolve()

    summary_path = split_dir / "summary.json"
    if not summary_path.exists():
        print(f"[verify] summary.json missing in {split_dir}", file=sys.stderr)
        return 2
    summary = json.loads(summary_path.read_text())

    manifests = read_manifests(split_dir)
    counts = {s: len(rs) for s, rs in manifests.items()}
    expected = summary.get("totals", {})
    for split in ("train", "val", "test"):
        e = int(expected.get(split, 0))
        a = counts.get(split, 0)
        assert e == a, f"{split}: summary={e} vs parquet={a}"
    print(f"[verify] counts OK: {counts}")

    # Speaker disjointness
    spk_to_splits: dict[str, set[str]] = defaultdict(set)
    for s, rows in manifests.items():
        for r in rows:
            spk = r.get("speaker_id")
            if spk:
                spk_to_splits[spk].add(s)
    overlap = {k: sorted(v) for k, v in spk_to_splits.items() if len(v) > 1}
    if overlap and summary.get("speaker_disjoint"):
        print(f"[verify][FAIL] speaker overlap across splits: {dict(list(overlap.items())[:10])}", file=sys.stderr)
        return 3
    print(f"[verify] speaker disjointness OK (total unique speakers={len(spk_to_splits)})")

    # Row-id disjointness
    row_to_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for s, rows in manifests.items():
        for r in rows:
            row_to_splits[(r["dataset"], r["row_id"])].add(s)
    row_overlap = {k: sorted(v) for k, v in row_to_splits.items() if len(v) > 1}
    if row_overlap:
        print(f"[verify][FAIL] (dataset,row_id) overlap: {dict(list(row_overlap.items())[:10])}", file=sys.stderr)
        return 4
    print(f"[verify] row_id disjointness OK (total unique rows={len(row_to_splits)})")

    # Ratio drift (informational only)
    per_ds = defaultdict(lambda: defaultdict(int))
    for s, rows in manifests.items():
        for r in rows:
            per_ds[r["dataset"]][s] += 1
    print("[verify] per-corpus counts:")
    for ds, by_split in sorted(per_ds.items()):
        total = sum(by_split.values())
        parts = ", ".join(f"{s}={by_split.get(s, 0)} ({by_split.get(s, 0) / max(total, 1):.1%})" for s in ("train", "val", "test"))
        print(f"  {ds}: total={total} {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
