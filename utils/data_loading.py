"""HF data loading and split-building logic for training.

Two manifests:

* ``SPLIT_MANIFEST_DIR=splits/<train_name>/`` — provides the **train** split.
* ``EVAL_MANIFEST_DIR=splits/<eval_name>/``  — provides the **val** and **test**
  splits, shared across every experiment. Defaults to
  ``splits/combined_eval_v1`` so cross-experiment numbers are directly
  comparable. The val/test parquets in the train manifest are ignored.

A speaker-leakage guard runs at startup: any speaker that appears in train and
in eval (val ∪ test) raises immediately.
"""

from __future__ import annotations

import os
from pathlib import Path

from datasets import DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login as hf_login

from utils.data import _count_label_presence
from utils.misc import _ALL_TASKS, _KAZEMO_TASKS, REPO_ROOT

SPLIT_MANIFEST_DIR = os.environ.get("SPLIT_MANIFEST_DIR", "").strip()
EVAL_MANIFEST_DIR = os.environ.get("EVAL_MANIFEST_DIR", "").strip() or "splits/combined_eval_v1"


def _hf_login() -> None:
    load_dotenv(REPO_ROOT / ".env")
    tok = os.environ.get("HF_TOKEN")
    if tok:
        try:
            hf_login(token=tok)
        except Exception:
            pass


def _resolve_split_manifest_dir(raw: str, *, label: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p = p.resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"{label} not found: {p}")
    if not (p / "config.yaml").exists():
        raise FileNotFoundError(f"{label} {p} missing config.yaml")
    parquets = [f for f in ("train.parquet", "val.parquet", "test.parquet") if (p / f).exists()]
    if not parquets:
        raise FileNotFoundError(f"{label} {p} has no train/val/test parquet files")
    return p


def _check_speaker_leakage(train_dir: Path, eval_dir: Path) -> None:
    """Hard-fail if any (dataset, speaker_id) appears in train and in eval val/test."""
    from splits.io import read_manifests

    train_rows = read_manifests(train_dir).get("train", [])
    eval_manifests = read_manifests(eval_dir)
    eval_rows = (eval_manifests.get("val", []) or []) + (eval_manifests.get("test", []) or [])

    def keys(rows):
        return {(r.get("dataset"), r.get("speaker_id")) for r in rows
                if r.get("speaker_id") and r.get("dataset")}

    overlap = keys(train_rows) & keys(eval_rows)
    if overlap:
        sample = sorted(overlap)[:5]
        raise RuntimeError(
            f"Speaker leakage: {len(overlap)} (dataset, speaker) pair(s) appear in "
            f"train ({train_dir}) and eval val/test ({eval_dir}). "
            f"Examples: {sample}. Rebuild splits with the same seed or use a "
            f"different EVAL_MANIFEST_DIR."
        )


def build_splits_from_manifests(train_dir: Path, eval_dir: Path):
    """Train from *train_dir*; val + test (merged + per-corpus) from *eval_dir*."""
    from splits.materialize import materialize_named_val, materialize_split

    _hf_login()
    train_dir = Path(train_dir)
    eval_dir = Path(eval_dir)
    print(f"[data] SPLIT_MANIFEST_DIR: {train_dir}")
    print(f"[data] EVAL_MANIFEST_DIR:  {eval_dir}")

    _check_speaker_leakage(train_dir, eval_dir)

    train_splits_map = materialize_split(train_dir)
    eval_splits_map = materialize_split(eval_dir)
    named_val_splits = materialize_named_val(eval_dir)

    train_split = train_splits_map.get("train")
    val_split = eval_splits_map.get("val")
    test_split = eval_splits_map.get("test")
    if train_split is None:
        raise RuntimeError(f"{train_dir} missing train split after materialize")
    if val_split is None:
        raise RuntimeError(f"{eval_dir} missing val split after materialize")

    named_val_tasks = {name: set(_ALL_TASKS) for name in named_val_splits}
    if "kazemo" in named_val_tasks:
        named_val_tasks["kazemo"] = set(_KAZEMO_TASKS)

    merged_hf = DatasetDict({"train": train_split, "val": val_split})
    if test_split is not None and len(test_split) > 0:
        merged_hf["test"] = test_split

    composition = {
        "mode": "split_manifest",
        "manifest_dir": str(train_dir),
        "eval_manifest_dir": str(eval_dir),
        "train_total": len(train_split),
        "val_total": len(val_split),
        "test_total": len(test_split) if test_split is not None else 0,
        "train_label_counts": _count_label_presence(train_split),
        "val_label_counts": _count_label_presence(val_split),
        "test_label_counts": _count_label_presence(test_split) if test_split is not None else {},
        "named_val_tasks": named_val_tasks,
    }
    for name, split in named_val_splits.items():
        composition[f"named_val_{name}_size"] = len(split)

    return merged_hf, train_split, val_split, named_val_splits, composition


def build_mixed_train_val_splits():
    """Main entry point for training: returns (merged_hf, train, val, named_val, composition)."""
    if not SPLIT_MANIFEST_DIR:
        raise RuntimeError(
            "SPLIT_MANIFEST_DIR is not set. Every training run must point to a "
            "splits/<name>/ directory built by scripts/build_splits.py."
        )
    train_dir = _resolve_split_manifest_dir(SPLIT_MANIFEST_DIR, label="SPLIT_MANIFEST_DIR")
    eval_dir = _resolve_split_manifest_dir(EVAL_MANIFEST_DIR, label="EVAL_MANIFEST_DIR")
    return build_splits_from_manifests(train_dir, eval_dir)
