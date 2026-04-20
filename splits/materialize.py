"""Materialize manifest parquet files back into HF Datasets.

Used by both:
  - training (multihead/utils.py::build_splits_from_manifest_dir) — no need to
    re-download audio beyond what's already in HF cache; we just `select` rows
    by index from the source HF split.
  - HF push (scripts/push_splits_to_hf.py) — same logic, but with audio bytes
    carried along so the pushed dataset is self-contained.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Audio, Dataset, DatasetDict, Features, Value, concatenate_datasets, load_dataset

from splits.io import read_manifests
from splits.schema import (
    ALL_SPLITS,
    DATASET_BATCH01,
    DATASET_BATCH02,
    DATASET_KAZEMO,
    SPLIT_VAL,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_split_config(split_dir: Path) -> dict[str, Any]:
    import yaml

    cfg_path = split_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml missing in {split_dir}")
    return yaml.safe_load(cfg_path.read_text()) or {}


def _resolve_cache_dir(raw: str | None, default: str) -> str:
    if not raw:
        return str(REPO_ROOT / default)
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return str(p)


def _load_hf_source(ds_name: str, ds_cfg: dict[str, Any]) -> Dataset:
    hf_id = ds_cfg["hf_id"]
    hf_split = ds_cfg.get("hf_split", "train")
    cache_dir = _resolve_cache_dir(ds_cfg.get("cache_dir"), f"data/{ds_name}")
    ds = load_dataset(hf_id, cache_dir=cache_dir)
    split = ds[hf_split] if hf_split in ds else ds[list(ds.keys())[0]]
    if "audio" in split.column_names:
        split = split.cast_column("audio", Audio(decode=False))
    return split


def _load_kazemo_source(ds_cfg: dict[str, Any]) -> Dataset:
    from loaders.kazemo.load_data import load_kazemotts

    cache_dir = _resolve_cache_dir(ds_cfg.get("cache_dir"), "data/kazemo")
    max_samples = ds_cfg.get("max_samples")
    ds = load_kazemotts(cache_dir=cache_dir, max_samples=max_samples)
    split = ds.get("train", ds[list(ds.keys())[0]])
    if "audio" in split.column_names:
        split = split.cast_column("audio", Audio(decode=False))
    return split


def _slice_by_source_index(src: Dataset, indices: list[int]) -> Dataset:
    if not indices:
        return src.select([])
    return src.select(indices)


def _canonical_features(reference: Dataset | None = None) -> Features:
    audio_feat = (
        reference.features["audio"]
        if reference is not None and "audio" in reference.features
        else Audio(decode=False)
    )
    return Features(
        {
            "audio": audio_feat,
            "emotion": Value("string"),
            "gender": Value("string"),
            "age_category": Value("string"),
        }
    )


def _ensure_labels_and_cast(split: Dataset, reference: Dataset | None = None) -> Dataset:
    needed = ("emotion", "gender", "age_category")
    missing = [c for c in needed if c not in split.column_names]
    if missing:
        def _inject(row):
            for c in missing:
                row[c] = None
            return row
        split = split.map(_inject, desc=f"inject_missing:{','.join(missing)}")
    keep = ["audio", "emotion", "gender", "age_category"]
    if "audio" not in split.column_names:
        raise RuntimeError("materialize: source split is missing 'audio' column")
    split = split.select_columns(keep)
    target = _canonical_features(reference)
    return split.cast(target)


def materialize_split(split_dir: Path) -> dict[str, Dataset]:
    """Return {split_name: Dataset(audio, emotion, gender, age_category)} by joining manifests to HF sources.

    All splits share the same canonical schema so they can be concatenated.
    """
    split_dir = Path(split_dir)
    cfg = _read_split_config(split_dir)
    corpora_cfg = cfg.get("corpora", {}) or {}
    manifests = read_manifests(split_dir)

    # Group manifest rows by (split, dataset) → list of source_index
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for split_name, rows in manifests.items():
        for r in rows:
            grouped[(split_name, r["dataset"])].append(int(r["source_index"]))

    datasets_used = sorted({ds for (_, ds) in grouped.keys()})
    sources: dict[str, Dataset] = {}
    for ds in datasets_used:
        ds_cfg = corpora_cfg.get(ds, {})
        if ds == DATASET_KAZEMO:
            sources[ds] = _load_kazemo_source(ds_cfg)
        else:
            sources[ds] = _load_hf_source(ds, ds_cfg)

    # Pick a reference split for feature-casting (first non-kazemo source preferred)
    ref = None
    for ds in (DATASET_BATCH01, DATASET_BATCH02, DATASET_KAZEMO):
        if ds in sources:
            ref = sources[ds]
            break

    out: dict[str, Dataset] = {}
    for split_name in ALL_SPLITS:
        parts: list[Dataset] = []
        # Preserve order: batch01, batch02, kazemo
        for ds in (DATASET_BATCH01, DATASET_BATCH02, DATASET_KAZEMO):
            key = (split_name, ds)
            if key not in grouped:
                continue
            indices = grouped[key]
            sliced = _slice_by_source_index(sources[ds], indices)
            sliced = _ensure_labels_and_cast(sliced, reference=ref)
            parts.append(sliced)
        if not parts:
            # Empty split — keep schema via reference
            if ref is not None:
                empty = _ensure_labels_and_cast(ref.select([]), reference=ref)
                out[split_name] = empty
            else:
                out[split_name] = Dataset.from_dict({"audio": [], "emotion": [], "gender": [], "age_category": []})
        else:
            out[split_name] = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    return out


def _materialize_per_corpus(split_dir: Path, split_name: str) -> dict[str, Dataset]:
    """Slice a named manifest split (val or test) per-corpus.

    Returns ``{"batch01": ds, "batch02": ds, "kazemo": ds}`` — only includes
    corpora that have at least one row in this split. Used for per-corpus
    reporting so e.g. batch01 vs batch02 accuracy can be compared apples-to-
    apples (concatenating them hides per-distribution quality differences).
    """
    split_dir = Path(split_dir)
    cfg = _read_split_config(split_dir)
    corpora_cfg = cfg.get("corpora", {}) or {}
    manifests = read_manifests(split_dir)

    rows = manifests.get(split_name, [])
    per_ds: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        per_ds[r["dataset"]].append(int(r["source_index"]))

    datasets_used = sorted(per_ds.keys())
    sources: dict[str, Dataset] = {}
    for ds in datasets_used:
        ds_cfg = corpora_cfg.get(ds, {})
        if ds == DATASET_KAZEMO:
            sources[ds] = _load_kazemo_source(ds_cfg)
        else:
            sources[ds] = _load_hf_source(ds, ds_cfg)

    ref = None
    for ds in (DATASET_BATCH01, DATASET_BATCH02, DATASET_KAZEMO):
        if ds in sources:
            ref = sources[ds]
            break

    named: dict[str, Dataset] = {}
    for ds in (DATASET_BATCH01, DATASET_BATCH02, DATASET_KAZEMO):
        if ds in sources and per_ds[ds]:
            sliced = _slice_by_source_index(sources[ds], per_ds[ds])
            named[ds] = _ensure_labels_and_cast(sliced, reference=ref)
    return named


def materialize_named_val(split_dir: Path) -> dict[str, Dataset]:
    """Per-corpus val slices: ``{"batch01", "batch02", "kazemo"}`` (present-only)."""
    return _materialize_per_corpus(split_dir, SPLIT_VAL)


def materialize_named_test(split_dir: Path) -> dict[str, Dataset]:
    """Per-corpus test slices: ``{"batch01", "batch02", "kazemo"}`` (present-only)."""
    from splits.schema import SPLIT_TEST
    return _materialize_per_corpus(split_dir, SPLIT_TEST)


def load_split_as_hf_dataset(split_dir: Path) -> DatasetDict:
    """Simple DatasetDict(train/val/test) — used by the HF push script."""
    splits = materialize_split(split_dir)
    return DatasetDict({k: v for k, v in splits.items() if v is not None})
