#!/usr/bin/env python3
"""Prefetch audio files referenced by the first N rows of an HF audio dataset.

For datasets like ``umutkkgz/tr-full-dataset`` that store audio as individual
.wav files referenced from a ``data.jsonl`` manifest, ``load_dataset`` with a
``train[:N]`` slice doesn't pre-download the audio — each row triggers a
separate HTTP fetch during iteration, which is download-bound and slow.

This script parses the manifest, collects the audio path pattern for the first
N rows, and snapshot_downloads them in parallel into the HF hub cache so the
next ``load_dataset`` + iterate run hits local disk.

Usage:
    python comparison/prefetch_hf.py umutkkgz/tr-full-dataset 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def _parse_audio_paths(manifest_path: Path, limit: int) -> list[str]:
    paths: list[str] = []
    with open(manifest_path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            row = json.loads(line)
            # Typical schemas: {"audio": "audio_000/file.wav"} or {"audio": {"path": "..."}}
            a = row.get("audio")
            if isinstance(a, dict):
                a = a.get("path") or a.get("file")
            if isinstance(a, str):
                paths.append(a)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo_id", help="HF dataset id, e.g. umutkkgz/tr-full-dataset")
    ap.add_argument("n", type=int, help="Number of leading rows to prefetch audio for")
    ap.add_argument("--manifest", default="data.jsonl",
                    help='Manifest filename inside the repo (default "data.jsonl")')
    ap.add_argument("--max-workers", type=int, default=16)
    args = ap.parse_args()

    print(f"[prefetch] {args.repo_id}: fetching manifest {args.manifest}", flush=True)
    manifest_local = hf_hub_download(
        repo_id=args.repo_id, repo_type="dataset", filename=args.manifest,
    )
    audio_paths = _parse_audio_paths(Path(manifest_local), args.n)
    if not audio_paths:
        print("[prefetch] no audio paths parsed from manifest — nothing to do", flush=True)
        sys.exit(1)

    # De-dup and convert to allow_patterns — snapshot_download is fast for
    # many patterns but slow row-by-row; batching as explicit patterns works.
    unique = sorted(set(audio_paths))
    print(f"[prefetch] {len(audio_paths)} rows → {len(unique)} unique files", flush=True)

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=unique,
        max_workers=args.max_workers,
    )
    print("[prefetch] done — re-run the comparison; iteration will hit disk cache now.",
          flush=True)


if __name__ == "__main__":
    main()
