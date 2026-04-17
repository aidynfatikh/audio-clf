#!/usr/bin/env python3
"""Diagnose why load_dataset('issai/KazEmoTTS') returns ~261k rows when the
paper reports 54k. Tells us if the text-builder fallback got triggered or if
the repo genuinely contains more data than the website suggests.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

# Make sure the repo root is on sys.path so we can import the shared loader.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("DATASETS_AUDIO_BACKEND", "soundfile")

from datasets import load_dataset  # noqa: E402


CACHE_DIR = REPO_ROOT / "data" / "kazemo"


def main() -> None:
    print(f"[inspect] cache_dir = {CACHE_DIR}")
    ds = load_dataset("issai/KazEmoTTS", cache_dir=str(CACHE_DIR))

    print("\n── splits & sizes ─────────────────────────────────────────────")
    for split_name, split in ds.items():
        print(f"  {split_name!r:<12s}  {len(split):>8d} rows   cols={split.column_names}")

    split = ds.get("train", ds[list(ds.keys())[0]])
    n = len(split)

    print("\n── first 3 rows (structured audio ⇒ H2, garbage text ⇒ H1) ────")
    for i in range(min(3, n)):
        row = split[i]
        preview = {
            k: (
                {kk: type(vv).__name__ for kk, vv in v.items()}
                if isinstance(v, dict)
                else (repr(v)[:80] + "…" if len(repr(v)) > 80 else repr(v))
            )
            for k, v in row.items()
        }
        print(f"  [{i}] {preview}")

    if "audio" not in split.column_names:
        print("\n  ⚠ No 'audio' column — text-builder almost certainly active (H1).")
        if "text" in split.column_names:
            samples = [split[i]["text"] for i in range(min(5, n))]
            print(f"  First 5 text rows: {samples}")
        return

    print("\n── unique-audio-path audit (H2 sanity check) ─────────────────")
    sample_size = min(n, 20_000)
    paths = []
    for i in range(sample_size):
        a = split[i]["audio"]
        if isinstance(a, dict) and "path" in a and a["path"]:
            paths.append(a["path"])
    uniq_in_sample = len(set(paths))
    print(f"  Scanned {sample_size} rows: {uniq_in_sample} unique audio paths")
    if uniq_in_sample < sample_size * 0.8:
        print("  ⇒ Dataset has many rows pointing at same file (looks like augmentation metadata).")
    else:
        print("  ⇒ Each row points at a distinct file — dataset is genuinely that big.")

    print("\n── speaker / emotion distribution (first 20k rows) ────────────")
    emos: Counter = Counter()
    speakers_from_path: Counter = Counter()
    for i in range(min(n, 20_000)):
        row = split[i]
        if row.get("emotion"):
            emos[str(row["emotion"])] += 1
        a = row.get("audio")
        p = a.get("path") if isinstance(a, dict) else None
        if p:
            stem = Path(p).name
            tok = stem.split("_", 1)[0] if "_" in stem else stem
            speakers_from_path[tok] += 1
    print(f"  emotions (sampled): {dict(emos)}")
    print(f"  speakers (sampled, from filename): {dict(speakers_from_path)}")


if __name__ == "__main__":
    main()
