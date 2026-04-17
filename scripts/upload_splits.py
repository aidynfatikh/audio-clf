#!/usr/bin/env python
"""Upload a materialized split to the Hugging Face Hub as a DatasetDict.

Usage:
  python scripts/upload_splits.py \
      --split-dir splits/b1_b2_kazemo_v1 \
      --repo-id 01gumano1d/audio-clf-splits-v1 [--private]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from huggingface_hub import login as hf_login

from splits.materialize import load_split_as_hf_dataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True, type=Path)
    ap.add_argument("--repo-id", required=True, help="HF dataset repo id, e.g. org/name")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("HF_TOKEN not set (.env)", file=sys.stderr)
        return 2
    hf_login(token=tok)

    dd = load_split_as_hf_dataset(args.split_dir.resolve())
    print("[push] built DatasetDict:")
    for k, v in dd.items():
        print(f"  {k}: {len(v)} rows; cols={v.column_names}")
    dd.push_to_hub(args.repo_id, private=args.private)
    print(f"[push] done → {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
