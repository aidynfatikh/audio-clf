#!/usr/bin/env python3
"""Export all runs from the configured W&B project as JSON.

Writes one <run_id>.json per run (config + summary + full history) and an
index.json with the compact summary list. Usage:

    python3 scripts/wandb_export.py                # all runs
    python3 scripts/wandb_export.py exp1-hubert    # filter: name substring
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

ENTITY = os.environ["WANDB_ENTITY"]
PROJECT = os.environ["WANDB_PROJECT"]
OUT_DIR = REPO_ROOT / "results" / "wandb_export"
OUT_DIR.mkdir(parents=True, exist_ok=True)

name_filter = sys.argv[1] if len(sys.argv) > 1 else None

api = wandb.Api(timeout=60)
runs = api.runs(f"{ENTITY}/{PROJECT}")

index: list[dict] = []
for run in runs:
    if name_filter and name_filter not in (run.name or ""):
        continue
    print(f"→ {run.name}  ({run.state}, {run.id})")
    history = run.history(pandas=False, samples=20_000)
    record = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "tags": list(run.tags or []),
        "group": run.group,
        "created_at": str(run.created_at),
        "runtime_s": run.summary.get("_runtime"),
        "config": {k: v for k, v in run.config.items() if not k.startswith("_")},
        "summary": {k: v for k, v in run.summary.items() if not k.startswith("_")},
        "history": history,
    }
    (OUT_DIR / f"{run.id}.json").write_text(
        json.dumps(record, indent=2, default=str)
    )
    index.append({k: record[k] for k in ("id", "name", "state", "tags", "summary", "created_at")})

(OUT_DIR / "index.json").write_text(json.dumps(index, indent=2, default=str))
print(f"\nWrote {len(index)} runs to {OUT_DIR}")
