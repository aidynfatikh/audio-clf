#!/usr/bin/env python3
"""Prefetch audio files referenced by the first N rows of an HF audio dataset.

For datasets like ``umutkkgz/tr-full-dataset`` that store audio as individual
.wav files referenced from a ``data.jsonl`` manifest, ``load_dataset`` with a
``train[:N]`` slice doesn't pre-download the audio — each row triggers a
separate HTTP fetch during iteration, which is download-bound and slow.

Manifest ``audio`` values can be either:
  * relative paths inside the dataset repo (``audio_000/foo.wav``), or
  * full HF URLs pointing at another repo
    (``https://huggingface.co/datasets/<repo>/resolve/<rev>/<path>``).

This module normalises both forms, then parallel-downloads each file into the
HF hub cache so the next ``load_dataset`` + iterate run hits local disk.
Idempotent — ``hf_hub_download`` skips files already present in the cache.

Library usage (from evaluate.py):
    from comparison.prefetch_hf import prefetch_hf_audio
    prefetch_hf_audio("umutkkgz/tr-full-dataset", n=5000)

CLI usage:
    python comparison/prefetch_hf.py umutkkgz/tr-full-dataset 5000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import unquote, urlparse

from huggingface_hub import hf_hub_download
from tqdm import tqdm


# Matches paths of the form /datasets/<owner>/<repo>/resolve/<revision>/<filename>
_HF_URL_RE = re.compile(
    r"^/datasets/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[^/]+)/(?P<path>.+)$"
)


def _parse_spec(raw: str, default_repo: str) -> tuple[str, str, str]:
    """Return (repo_id, revision, filename) for either a full HF URL or a
    relative path within ``default_repo``. Non-HF URLs raise ValueError."""
    if raw.startswith(("http://", "https://")):
        parsed = urlparse(raw)
        if "huggingface.co" not in parsed.netloc:
            raise ValueError(f"unsupported non-HF URL: {raw}")
        m = _HF_URL_RE.match(parsed.path)
        if not m:
            raise ValueError(f"unrecognised HF URL shape: {raw}")
        return m["repo"], m["rev"], unquote(m["path"])
    return default_repo, "main", raw


def _parse_audio_specs(
    manifest_path: Path, limit: int | None, default_repo: str,
) -> list[tuple[str, str, str]]:
    specs: list[tuple[str, str, str]] = []
    with open(manifest_path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            # Schemas: {"audio": "path.wav"} | {"audio": {"path": "..."}}
            a = row.get("audio")
            if isinstance(a, dict):
                a = a.get("path") or a.get("file") or a.get("url")
            if not isinstance(a, str):
                continue
            try:
                specs.append(_parse_spec(a, default_repo))
            except ValueError:
                continue
    return specs


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
    specs = _parse_audio_specs(Path(manifest_local), n, repo_id)
    if not specs:
        _log("[prefetch] no audio paths parsed from manifest — nothing to do")
        return 0

    unique = sorted(set(specs))
    source_repos = sorted({r for r, _, _ in unique})
    _log(
        f"[prefetch] {len(specs)} rows → {len(unique)} unique files "
        f"across {len(source_repos)} repo(s): {source_repos}"
    )

    def _fetch(spec: tuple[str, str, str]) -> None:
        repo, rev, path = spec
        hf_hub_download(
            repo_id=repo, repo_type="dataset", filename=path, revision=rev,
        )

    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, s): s for s in unique}
        iterator = as_completed(futures)
        if not quiet:
            iterator = tqdm(iterator, total=len(futures), unit="file")
        for fut in iterator:
            try:
                fut.result()
            except Exception as e:
                errors += 1
                if errors <= 5:
                    repo, rev, path = futures[fut]
                    _log(f"[prefetch] failed {repo}@{rev}:{path}: {e}")

    if errors:
        raise RuntimeError(
            f"{errors}/{len(unique)} files failed to download — "
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
