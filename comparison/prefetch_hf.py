#!/usr/bin/env python3
"""Prefetch audio files referenced by an HF audio-manifest dataset.

Datasets like ``umutkkgz/tr-full-dataset`` ship a ``data.jsonl`` manifest that
points at individual .wav files — either as relative paths inside the same
repo, or as full ``https://huggingface.co/datasets/<repo>/resolve/<rev>/...``
URLs into another repo. ``load_dataset`` + iterate streams these one-by-one
over HTTP, which is painfully slow.

This module parses the manifest, downloads every referenced file in parallel
into the HF hub cache, and returns the manifest rows with ``audio`` rewritten
to the **local cached path** so downstream code can build a ``Dataset`` that
reads purely from disk.

Library usage (from evaluate.py):
    rows = prefetch_hf_audio("umutkkgz/tr-full-dataset", n=5000)
    # rows = [{"audio": "/home/.../segment_000.wav", "emotion": "happy", ...}, ...]

CLI usage (smoke test — just reports the count):
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


# /datasets/<owner>/<repo>/resolve/<revision>/<filename>
_HF_URL_RE = re.compile(
    r"^/datasets/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[^/]+)/(?P<path>.+)$"
)


def _parse_spec(raw: str, default_repo: str) -> tuple[str, str, str]:
    """(repo_id, revision, filename) for a full HF URL or a relative path."""
    if raw.startswith(("http://", "https://")):
        parsed = urlparse(raw)
        if "huggingface.co" not in parsed.netloc:
            raise ValueError(f"unsupported non-HF URL: {raw}")
        m = _HF_URL_RE.match(parsed.path)
        if not m:
            raise ValueError(f"unrecognised HF URL shape: {raw}")
        return m["repo"], m["rev"], unquote(m["path"])
    return default_repo, "main", raw


def _read_manifest(manifest_path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with open(manifest_path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def _row_audio_raw(row: dict) -> str | None:
    a = row.get("audio")
    if isinstance(a, dict):
        a = a.get("path") or a.get("file") or a.get("url")
    return a if isinstance(a, str) else None


def prefetch_hf_audio(
    repo_id: str,
    n: int | None,
    manifest: str = "data.jsonl",
    max_workers: int = 16,
    quiet: bool = False,
) -> list[dict]:
    """Download the first ``n`` manifest rows' audio and return localized rows.

    Each returned row is the original manifest row with ``audio`` replaced by
    the absolute path of the cached .wav file. Rows whose audio field can't be
    parsed are dropped. Raises ``RuntimeError`` if any download fails.
    """
    def _log(msg: str) -> None:
        if not quiet:
            print(msg, flush=True)

    _log(f"[prefetch] {repo_id}: fetching manifest {manifest}")
    manifest_local = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=manifest,
    )
    raw_rows = _read_manifest(Path(manifest_local), n)

    specs: list[tuple[str, str, str] | None] = []
    for row in raw_rows:
        raw = _row_audio_raw(row)
        if raw is None:
            specs.append(None)
            continue
        try:
            specs.append(_parse_spec(raw, repo_id))
        except ValueError as e:
            _log(f"[prefetch] skip row: {e}")
            specs.append(None)

    unique = sorted({s for s in specs if s is not None})
    if not unique:
        _log("[prefetch] no usable audio paths in manifest")
        return []

    source_repos = sorted({r for r, _, _ in unique})
    _log(
        f"[prefetch] {len(raw_rows)} rows → {len(unique)} unique files "
        f"across {len(source_repos)} repo(s): {source_repos}"
    )

    local_paths: dict[tuple[str, str, str], str] = {}

    def _fetch(spec: tuple[str, str, str]) -> tuple[tuple[str, str, str], str]:
        repo, rev, path = spec
        local = hf_hub_download(
            repo_id=repo, repo_type="dataset", filename=path, revision=rev,
        )
        return spec, local

    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, s): s for s in unique}
        iterator = as_completed(futures)
        if not quiet:
            iterator = tqdm(iterator, total=len(futures), unit="file")
        for fut in iterator:
            try:
                spec, local = fut.result()
                local_paths[spec] = local
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

    localized: list[dict] = []
    for row, spec in zip(raw_rows, specs):
        if spec is None:
            continue
        new_row = dict(row)
        new_row["audio"] = local_paths[spec]
        localized.append(new_row)

    _log(f"[prefetch] done — {len(localized)} rows ready for local iteration.")
    return localized


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo_id", help="HF dataset id, e.g. umutkkgz/tr-full-dataset")
    ap.add_argument("n", type=int, help="Number of leading rows to prefetch audio for")
    ap.add_argument("--manifest", default="data.jsonl",
                    help='Manifest filename inside the repo (default "data.jsonl")')
    ap.add_argument("--max-workers", type=int, default=16)
    args = ap.parse_args()

    try:
        rows = prefetch_hf_audio(
            args.repo_id, args.n,
            manifest=args.manifest, max_workers=args.max_workers,
        )
    except RuntimeError as e:
        print(f"[prefetch] {e}", flush=True)
        sys.exit(1)
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
