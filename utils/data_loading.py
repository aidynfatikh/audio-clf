"""HF data loading and split-building logic for training.

Controls via env vars:
  SPLIT_MANIFEST_DIR  — path to a splits/<name>/ directory; takes precedence over all below
  TRAIN_VAL_MANIFEST  — legacy: path to the pre-baked 1008-sample batch01 holdout JSON
  USE_BATCH02         — include batch02 in training (default: 1)
  USE_BATCH01_TRAIN   — include batch01 in training (default: 1)
  USE_KAZEMO          — include KazEmoTTS in training (default: 1)
  KAZEMO_MAX_SAMPLES  — cap on KazEmoTTS samples (default: 20000)
  KAZEMO_VAL_FRACTION — fraction of KazEmoTTS held out for val in legacy mode (default: 0.1)
  HF_BATCH01_ID / HF_BATCH02_ID  — HF dataset identifiers
  HF_BATCH01_SPLIT    — which split of batch01 to use (default: train)
  HF_BATCH01_CACHE / HF_BATCH02_CACHE — local cache paths
"""

from __future__ import annotations

import os
from pathlib import Path

from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login as hf_login

from eval.validation_holdout import (
    first_index_by_row_id,
    holdout_source_id_set,
    load_validation_sample_ids,
    train_indices_excluding_holdout,
    val_indices_from_manifest,
)
from loaders.kazemo.load_data import load_kazemotts
from loaders.load_data import DATA_DIR, load
from utils.data import (
    _count_emotion_distribution,
    _count_label_presence,
    _force_canonical_label_schema,
    _prepare_split_for_training,
    VAL_FRACTION,
    fallback_split_train_val,
)
from utils.misc import _ALL_TASKS, _KAZEMO_TASKS, RANDOM_SEED, REPO_ROOT

# ── Env vars ─────────────────────────────────────────────────────────────────
SPLIT_MANIFEST_DIR = os.environ.get("SPLIT_MANIFEST_DIR", "").strip()
TRAIN_VAL_MANIFEST = os.environ.get("TRAIN_VAL_MANIFEST", "").strip()
HF_BATCH01_ID = os.environ.get("HF_BATCH01_ID", "01gumano1d/batch01-validation-test")
HF_BATCH02_ID = os.environ.get("HF_BATCH02_ID", "01gumano1d/batch2-aug-clean")
USE_BATCH02 = os.environ.get("USE_BATCH02", "1").strip().lower() not in {"0", "false", "no"}
USE_BATCH01_TRAIN = os.environ.get("USE_BATCH01_TRAIN", "1").strip().lower() not in {"0", "false", "no"}
USE_KAZEMO = os.environ.get("USE_KAZEMO", "1").strip().lower() not in {"0", "false", "no"}
KAZEMO_MAX_SAMPLES = int(os.environ.get("KAZEMO_MAX_SAMPLES", "20000"))
KAZEMO_VAL_FRACTION = float(os.environ.get("KAZEMO_VAL_FRACTION", "0.1"))
HF_BATCH01_SPLIT = os.environ.get("HF_BATCH01_SPLIT", "train")
HF_BATCH01_CACHE = Path(
    os.environ.get("HF_BATCH01_CACHE", str(REPO_ROOT / "data" / "batch01-validation-test"))
)
HF_BATCH02_CACHE = Path(
    os.environ.get("HF_BATCH02_CACHE", str(REPO_ROOT / "data" / "batch2-aug-clean"))
)


def _hf_login() -> None:
    load_dotenv(REPO_ROOT / ".env")
    tok = os.environ.get("HF_TOKEN")
    if tok:
        try:
            hf_login(token=tok)
        except Exception:
            pass


# ── Manifest-dir path (new split builder) ───────────────────────────────────

def _resolve_split_manifest_dir(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p = p.resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"SPLIT_MANIFEST_DIR not found: {p}")
    if not (p / "config.yaml").exists():
        raise FileNotFoundError(f"SPLIT_MANIFEST_DIR {p} missing config.yaml")
    parquets = [f for f in ("train.parquet", "val.parquet", "test.parquet") if (p / f).exists()]
    if not parquets:
        raise FileNotFoundError(f"SPLIT_MANIFEST_DIR {p} has no train/val/test parquet files")
    return p


def build_splits_from_manifest_dir(split_dir: Path):
    """Load a splits/<name>/ directory and return the legacy training 5-tuple."""
    from splits.materialize import materialize_named_val, materialize_split

    _hf_login()
    split_dir = Path(split_dir)
    print(f"[data] SPLIT_MANIFEST_DIR: {split_dir}")

    splits_map = materialize_split(split_dir)
    named_val_splits = materialize_named_val(split_dir)

    train_split = splits_map.get("train")
    val_split = splits_map.get("val")
    test_split = splits_map.get("test")
    if train_split is None or val_split is None:
        raise RuntimeError(f"SPLIT_MANIFEST_DIR {split_dir} missing train/val after materialize")

    named_val_tasks = {name: set(_ALL_TASKS) for name in named_val_splits}
    if "kazemo" in named_val_tasks:
        named_val_tasks["kazemo"] = set(_KAZEMO_TASKS)

    merged_hf = DatasetDict({"train": train_split, "val": val_split})
    if test_split is not None and len(test_split) > 0:
        merged_hf["test"] = test_split

    composition = {
        "mode": "split_manifest",
        "manifest_dir": str(split_dir),
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


# ── Legacy holdout-manifest path ─────────────────────────────────────────────

def _resolve_train_val_manifest_path(raw: str) -> Path:
    mp = Path(raw)
    if not mp.is_absolute():
        mp = REPO_ROOT / mp
    resolved = mp.resolve()
    if resolved.exists():
        return resolved
    for alt in (REPO_ROOT / "results" / mp.name, REPO_ROOT / "eval" / mp.name):
        if alt.exists():
            print(f"[data] Holdout manifest: using {alt} (configured path missing: {resolved})")
            return alt.resolve()
    raise FileNotFoundError(
        f"Holdout manifest not found. Tried {resolved}, "
        f"{REPO_ROOT / 'results' / mp.name}, {REPO_ROOT / 'eval' / mp.name}. "
        "Run eval/validate.py (writes under results/) or set TRAIN_VAL_MANIFEST."
    )


def build_holdout_mixed_train_val_splits(manifest_path: Path):
    _hf_login()
    manifest_path = manifest_path.resolve()
    sample_ids = load_validation_sample_ids(manifest_path)
    holdout_bases = holdout_source_id_set(sample_ids)

    print(f"[data] Holdout manifest: {manifest_path} ({len(sample_ids)} validation rows)")
    HF_BATCH01_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"[data] batch01: {HF_BATCH01_ID!r} cache={HF_BATCH01_CACHE} split={HF_BATCH01_SPLIT!r}")
    ds1 = load_dataset(HF_BATCH01_ID, cache_dir=str(HF_BATCH01_CACHE))
    split1 = ds1[HF_BATCH01_SPLIT] if HF_BATCH01_SPLIT in ds1 else ds1[list(ds1.keys())[0]]
    split1 = _force_canonical_label_schema(_prepare_split_for_training(split1))

    id_to_idx = first_index_by_row_id(split1)
    val_indices = val_indices_from_manifest(sample_ids, id_to_idx)
    val_split = split1.select(val_indices)
    if len(val_split) != len(sample_ids):
        raise RuntimeError(f"Val size {len(val_split)} != manifest {len(sample_ids)}")

    train_idx = train_indices_excluding_holdout(split1, holdout_bases)
    train_b1 = split1.select(train_idx)

    if not USE_BATCH01_TRAIN:
        print("[data] USE_BATCH01_TRAIN=0: excluding batch01 from train.")
        train_b1 = split1.select([])

    if USE_BATCH02:
        HF_BATCH02_CACHE.mkdir(parents=True, exist_ok=True)
        print(f"[data] batch02 → train only: {HF_BATCH02_ID!r} cache={HF_BATCH02_CACHE}")
        ds2 = load_dataset(HF_BATCH02_ID, cache_dir=str(HF_BATCH02_CACHE))
        b2 = ds2.get("train") or ds2[list(ds2.keys())[0]]
        b2 = _force_canonical_label_schema(_prepare_split_for_training(b2), reference_split=train_b1)
    else:
        print("[data] USE_BATCH02=0: skipping batch02.")
        b2 = train_b1.select([])

    kazemo_train_count = 0
    kazemo_val_count = 0
    kazemo_emotion_counts: dict = {}

    holdout_val = val_split
    train_split = concatenate_datasets([train_b1, b2]) if len(b2) > 0 else train_b1
    named_val_splits: dict = {}

    if USE_KAZEMO:
        print(f"[data] USE_KAZEMO=1: appending KazEmo (cap={KAZEMO_MAX_SAMPLES})")
        kz_ds: DatasetDict = load_kazemotts(cache_dir=str(DATA_DIR), max_samples=KAZEMO_MAX_SAMPLES)
        kz_base = _prepare_split_for_training(kz_ds.get("train", kz_ds[list(kz_ds.keys())[0]]))
        kz_split = kz_base.train_test_split(
            test_size=KAZEMO_VAL_FRACTION, seed=RANDOM_SEED, shuffle=True
        )
        kz_train = _force_canonical_label_schema(
            _prepare_split_for_training(kz_split["train"]), reference_split=train_b1
        )
        kz_val = _force_canonical_label_schema(
            _prepare_split_for_training(kz_split["test"]), reference_split=holdout_val
        )
        kazemo_train_count = len(kz_train)
        kazemo_val_count = len(kz_val)
        kazemo_emotion_counts = _count_emotion_distribution(kz_base)
        train_split = concatenate_datasets([train_split, kz_train])
        val_split = concatenate_datasets([holdout_val, kz_val])
        named_val_splits = {"val": holdout_val, "kazemo": kz_val}
    else:
        named_val_splits = {"val": holdout_val}

    named_val_tasks = {name: set(_ALL_TASKS) for name in named_val_splits}
    if "kazemo" in named_val_tasks:
        named_val_tasks["kazemo"] = set(_KAZEMO_TASKS)

    composition = {
        "mode": "holdout_manifest",
        "manifest": str(manifest_path),
        "batch01_id": HF_BATCH01_ID,
        "batch02_id": HF_BATCH02_ID,
        "use_batch02": USE_BATCH02,
        "use_batch01_train": USE_BATCH01_TRAIN,
        "batch01_split": HF_BATCH01_SPLIT,
        "batch01_train_only": len(train_b1),
        "batch02_train_only": len(b2),
        "hf_train": len(train_b1),
        "hf_val": len(holdout_val),
        "kazemo_train": kazemo_train_count,
        "kazemo_val": kazemo_val_count,
        "kazemo_emotion_counts": kazemo_emotion_counts,
        "train_total": len(train_split),
        "val_total": len(val_split),
        "train_label_counts": _count_label_presence(train_split),
        "val_label_counts": _count_label_presence(val_split),
        "named_val_tasks": named_val_tasks,
    }
    merged_hf = DatasetDict(
        {"batch01_train": train_b1, "batch01_val_holdout": holdout_val, "batch02_train": b2}
    )
    return merged_hf, train_split, val_split, named_val_splits, composition


def build_mixed_train_val_splits():
    """Main entry point for training: returns the legacy 5-tuple (hf_ds, train, val, named, comp)."""
    if SPLIT_MANIFEST_DIR:
        return build_splits_from_manifest_dir(_resolve_split_manifest_dir(SPLIT_MANIFEST_DIR))
    if TRAIN_VAL_MANIFEST:
        return build_holdout_mixed_train_val_splits(
            _resolve_train_val_manifest_path(TRAIN_VAL_MANIFEST)
        )

    _hf_login()
    hf_dataset = load()
    hf_train = hf_dataset.get("train") or hf_dataset[list(hf_dataset.keys())[0]]
    hf_val = hf_dataset.get("validation", hf_dataset.get("val", hf_dataset.get("test")))
    if hf_val is None:
        print(f"Warning: no validation split; splitting train with val_fraction={VAL_FRACTION}.")
        hf_train, hf_val = fallback_split_train_val(hf_train, seed=RANDOM_SEED, val_fraction=VAL_FRACTION)

    hf_train = _force_canonical_label_schema(_prepare_split_for_training(hf_train))
    hf_val = _force_canonical_label_schema(_prepare_split_for_training(hf_val))

    train_split = hf_train
    val_split = hf_val
    kazemo_train_count = 0
    kazemo_val_count = 0
    kazemo_emotion_counts: dict = {}
    named_val_splits: dict = {}

    if USE_KAZEMO:
        print(f"[data] USE_KAZEMO=1: appending KazEmo (cap={KAZEMO_MAX_SAMPLES})")
        kz_ds: DatasetDict = load_kazemotts(cache_dir=str(DATA_DIR), max_samples=KAZEMO_MAX_SAMPLES)
        kz_base = _prepare_split_for_training(kz_ds.get("train", kz_ds[list(kz_ds.keys())[0]]))
        kz_split = kz_base.train_test_split(
            test_size=KAZEMO_VAL_FRACTION, seed=RANDOM_SEED, shuffle=True
        )
        kz_train = _force_canonical_label_schema(
            _prepare_split_for_training(kz_split["train"]), reference_split=hf_train
        )
        kz_val = _force_canonical_label_schema(
            _prepare_split_for_training(kz_split["test"]), reference_split=hf_val
        )
        kazemo_train_count = len(kz_train)
        kazemo_val_count = len(kz_val)
        kazemo_emotion_counts = _count_emotion_distribution(kz_base)
        train_split = concatenate_datasets([hf_train, kz_train])
        val_split = concatenate_datasets([hf_val, kz_val])
        named_val_splits = {"val": hf_val, "kazemo": kz_val}
    else:
        named_val_splits = {"val": val_split}

    named_val_tasks = {name: set(_ALL_TASKS) for name in named_val_splits}
    if "kazemo" in named_val_tasks:
        named_val_tasks["kazemo"] = set(_KAZEMO_TASKS)

    composition = {
        "hf_train": len(hf_train),
        "hf_val": len(hf_val),
        "kazemo_train": kazemo_train_count,
        "kazemo_val": kazemo_val_count,
        "kazemo_emotion_counts": kazemo_emotion_counts,
        "train_total": len(train_split),
        "val_total": len(val_split),
        "train_label_counts": _count_label_presence(train_split),
        "val_label_counts": _count_label_presence(val_split),
        "named_val_tasks": named_val_tasks,
    }
    return hf_dataset, train_split, val_split, named_val_splits, composition
