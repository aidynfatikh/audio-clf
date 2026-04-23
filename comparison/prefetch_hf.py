#!/usr/bin/env python3
"""Prefetch audio files referenced by the first N rows of an HF audio dataset.

For datasets like ``umutkkgz/tr-full-dataset`` that store audio as individual
.wav files referenced from a ``data.jsonl`` manifest, ``load_dataset`` with a
``train[:N]`` slice doesn't pre-download the audio — each row triggers a
separate HTTP fetch during iteration, which is download-bound and slow.

This module parses the manifest, collects the audio path pattern for the first
N rows, and parallel-downloads them into the HF hub cache so the next
``load_dataset`` + iterate run hits local disk. Idempotent — hf_hub_download
skips files already present in the cache.

Library usage (from evaluate.py):
    from comparison.prefetch_hf import prefetch_hf_audio
    prefetch_hf_audio("umutkkgz/tr-full-dataset", n=5000)

CLI usage:
    python comparison/prefetch_hf.py umutkkgz/tr-full-dataset 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm


def _parse_audio_paths(manifest_path: Path, limit: int | None) -> list[str]:
    paths: list[str] = []
    with open(manifest_path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            # Typical schemas: {"audio": "audio_000/file.wav"} or {"audio": {"path": "..."}}
            a = row.get("audio")
            if isinstance(a, dict):
                a = a.get("path") or a.get("file")
            if isinstance(a, str):
                paths.append(a)
    return paths


def prefetch_hf_audio(
    repo_id: str,
    n: int | None,
    manifest: str = "data.jsonl",
    max_workers: int = 16,
    quiet: bool = False,
) -> int:
    """Prefetch audio files referenced by the first ``n`` manifest rows.

    Returns the number of unique files fetched. Raises on missing manifest so
    callers can decide whether to fall back (evaluate.py treats it as a warning).
    """
    def _log(msg: str) -> None:
        if not quiet:
            print(msg, flush=True)

    _log(f"[prefetch] {repo_id}: fetching manifest {manifest}")
    manifest_local = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=manifest,
    )
    audio_paths = _parse_audio_paths(Path(manifest_local), n)
    if not audio_paths:
        _log("[prefetch] no audio paths parsed from manifest — nothing to do")
        return 0

    unique = sorted(set(audio_paths))
    _log(f"[prefetch] {len(audio_paths)} rows → {len(unique)} unique files")

    def _fetch(path: str) -> None:
        hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=path)

    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, p): p for p in unique}
        iterator = as_completed(futures)
        if not quiet:
            iterator = tqdm(iterator, total=len(futures), unit="file")
        for fut in iterator:
            try:
                fut.result()
            except Exception as e:
                errors += 1
                if errors <= 5:
                    _log(f"[prefetch] failed {futures[fut]}: {e}")

    if errors:
        raise RuntimeError(
            f"{errors} files failed to download from {repo_id} — "
            "retry or reduce max_workers"
        )
    _log("[prefetch] done — iteration will hit disk cache now.")
    return len(unique)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo_id", help="HF dataset id, e.g. umutkkgz/tr-full-dataset")
    ap.add_argument("n", type=int, help="Number of leading rows to prefetch audio for")
    ap.add_argument("--manifest", default="data.jsonl",
                    help='Manifest filename inside the repo (default "data.jsonl")')
    ap.add_argument("--max-workers", type=int, default=16)
    args = ap.parse_args()

    try:
        count = prefetch_hf_audio(
            args.repo_id, args.n,
            manifest=args.manifest, max_workers=args.max_workers,
        )
    except RuntimeError as e:
        print(f"[prefetch] {e}", flush=True)
        sys.exit(1)
    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
