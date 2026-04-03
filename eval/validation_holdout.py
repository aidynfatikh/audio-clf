"""Shared helpers for the fixed 1008-example validation holdout from validate.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent


def base_row_id(sid: str) -> str:
    """Dataset clip id as in HF row['id'] (strip synthetic repad suffix from validate.py)."""
    if "__repad_" in sid:
        return sid.split("__repad_")[0]
    return sid


def row_hf_id(row: dict[str, Any], idx: int) -> str:
    """Match validate.py / train row identity."""
    return str(row.get("id", row.get("uid", f"row-{idx}")))


def load_validation_sample_ids(manifest_path: Path | str) -> list[str]:
    path = Path(manifest_path)
    with open(path) as f:
        data = json.load(f)
    ids = data.get("validation_sample_ids")
    if isinstance(ids, list) and ids:
        return [str(x) for x in ids]
    # Older manifests: same order as validate.py eval report (sibling or repo results/)
    alt_candidates = [
        path.parent / "validation_eval_results.json",
        _REPO_ROOT / "results" / "validation_eval_results.json",
    ]
    for alt in alt_candidates:
        if alt.exists():
            with open(alt) as f:
                ev = json.load(f)
            samples = ev.get("samples") or []
            out = [str(s.get("sample_id", "")) for s in samples if s.get("sample_id")]
            n = int(ev.get("total_samples") or 0)
            if n and len(out) == n:
                return out
            if not n and out:
                return out
    raise ValueError(
        f"{path} has no validation_sample_ids; run validate.py again, "
        "or keep results/validation_eval_results.json with samples[].sample_id (same order as eval)."
    )


def holdout_source_id_set(sample_ids: list[str]) -> set[str]:
    return {base_row_id(s) for s in sample_ids}


def first_index_by_row_id(dataset) -> dict[str, int]:
    """First occurrence index for each row_hf_id (same rule as row index in split order)."""
    out: dict[str, int] = {}
    n = len(dataset)
    for i in range(n):
        rid = row_hf_id(dataset[i], i)
        if rid not in out:
            out[rid] = i
    return out


def val_indices_from_manifest(sample_ids: list[str], id_to_idx: dict[str, int]) -> list[int]:
    out: list[int] = []
    for sid in sample_ids:
        base = base_row_id(sid)
        if base not in id_to_idx:
            raise ValueError(
                f"Holdout sample id {base!r} (from manifest entry {sid!r}) not found in batch01 split."
            )
        out.append(id_to_idx[base])
    return out


def train_indices_excluding_holdout(dataset, holdout_bases: set[str]) -> list[int]:
    n = len(dataset)
    return [i for i in range(n) if row_hf_id(dataset[i], i) not in holdout_bases]
