"""Corpus loaders that produce NormalizedRow lists.

No audio bytes are kept here — only the metadata needed to compute splits and
later join back to the source HF datasets via (dataset, row_id).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Iterable

from datasets import Audio, load_dataset
from tqdm import tqdm

from splits.augmented_filter import extract_augmented_flag
from splits.kazemo_speakers import kazemo_filename_stem, parse_kazemo_speaker
from splits.schema import (
    DATASET_BATCH01,
    DATASET_BATCH02,
    DATASET_KAZEMO,
    NormalizedRow,
)


def _str_or_none(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _row_id_for_hf(row: dict[str, Any], idx: int) -> str:
    """Match eval/validation_holdout.row_hf_id — always returns str."""
    return str(row.get("id", row.get("uid", f"row-{idx}")))


def _load_hf_metadata_split(hf_id: str, hf_split: str, cache_dir: Path):
    """Load the configured split and strip audio decoding (we only need metadata)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(hf_id, cache_dir=str(cache_dir))
    if hf_split in ds:
        split = ds[hf_split]
    else:
        split = ds[list(ds.keys())[0]]
    if "audio" in split.column_names:
        split = split.cast_column("audio", Audio(decode=False))
    return split


def _load_hf_corpus(
    dataset_name: str,
    *,
    hf_id: str,
    hf_split: str,
    cache_dir: Path,
    speaker_column: str | None,
    drop_augmented: bool,
) -> list[NormalizedRow]:
    """Load a HF corpus → list[NormalizedRow].

    `drop_augmented=True` removes aug rows at load time (equivalent to the old
    `filter_augmented: true` / new `augmented_policy: drop_all`). For any other
    policy we return every row with the `augmented` flag populated; the
    builder decides per-split routing.
    """
    split = _load_hf_metadata_split(hf_id, hf_split, cache_dir)
    columns = set(split.column_names)
    has_augmented_col = "augmented" in columns or "is_augmented" in columns or "metadata" in columns
    has_speaker = speaker_column is not None and speaker_column in columns

    # Iterate without decoding audio — much faster on large corpora.
    drop_audio = [c for c in ("audio",) if c in columns]
    meta_split = split.remove_columns(drop_audio) if drop_audio else split

    rows: list[NormalizedRow] = []
    aug_true = 0
    aug_false = 0
    dropped_aug = 0
    for idx, row in enumerate(tqdm(meta_split, desc=f"scan {dataset_name}", leave=False)):
        aug = extract_augmented_flag(row) if has_augmented_col else None
        if aug is True:
            aug_true += 1
        elif aug is False:
            aug_false += 1
        if drop_augmented and aug is True:
            dropped_aug += 1
            continue
        rows.append(
            NormalizedRow(
                dataset=dataset_name,
                row_id=_row_id_for_hf(row, idx),
                source_index=idx,
                speaker_id=_str_or_none(row.get(speaker_column)) if has_speaker else None,
                emotion=_str_or_none(row.get("emotion")),
                gender=_str_or_none(row.get("gender")),
                age_category=_str_or_none(row.get("age_category")),
                augmented=aug,
            )
        )
    print(
        f"[sources] {dataset_name}: kept {len(rows)} rows "
        f"(aug_true={aug_true}, aug_false={aug_false}, dropped_at_source={dropped_aug}); "
        f"speaker_column_present={has_speaker}, augmented_col_present={has_augmented_col}"
    )
    return rows


def _should_drop_at_source(cfg: dict[str, Any]) -> bool:
    """Only the `drop_all` policy filters at load time; everything else keeps
    augmented rows so the builder can route them per split.
    """
    policy = _resolve_aug_policy(cfg)
    return policy == "drop_all"


def _resolve_aug_policy(cfg: dict[str, Any]) -> str:
    """Read `augmented_policy` with back-compat for legacy `filter_augmented`."""
    raw = cfg.get("augmented_policy")
    if raw:
        p = str(raw).strip().lower()
        if p not in {"drop_all", "keep_all", "train_only"}:
            raise ValueError(f"augmented_policy must be drop_all | keep_all | train_only, got {raw!r}")
        return p
    if "filter_augmented" in cfg:
        return "drop_all" if bool(cfg["filter_augmented"]) else "keep_all"
    return "train_only"  # sensible default: keep aug in train, clean eval


def load_batch01(cfg: dict[str, Any]) -> list[NormalizedRow]:
    return _load_hf_corpus(
        DATASET_BATCH01,
        hf_id=cfg["hf_id"],
        hf_split=cfg.get("hf_split", "train"),
        cache_dir=Path(cfg["cache_dir"]),
        speaker_column=cfg.get("speaker_column", "speaker_id"),
        drop_augmented=_should_drop_at_source(cfg),
    )


def load_batch02(cfg: dict[str, Any]) -> list[NormalizedRow]:
    return _load_hf_corpus(
        DATASET_BATCH02,
        hf_id=cfg["hf_id"],
        hf_split=cfg.get("hf_split", "train"),
        cache_dir=Path(cfg["cache_dir"]),
        speaker_column=cfg.get("speaker_column", "speaker_id"),
        drop_augmented=_should_drop_at_source(cfg),
    )


def load_kazemo(cfg: dict[str, Any]) -> list[NormalizedRow]:
    """Scan KazEmoTTS, parse speaker from filename, emit NormalizedRow list."""
    from loaders.kazemo.load_data import load_kazemotts

    cache_dir = cfg.get("cache_dir")
    max_samples = cfg.get("max_samples")
    if isinstance(max_samples, str) and not max_samples.strip():
        max_samples = None

    ds = load_kazemotts(cache_dir=str(cache_dir) if cache_dir else None, max_samples=max_samples)
    split = ds.get("train", ds[list(ds.keys())[0]])
    if "audio" in split.column_names:
        split = split.cast_column("audio", Audio(decode=False))

    rows: list[NormalizedRow] = []
    missing_speakers = 0
    for idx, row in enumerate(tqdm(split, desc="scan kazemo", leave=False)):
        speaker = parse_kazemo_speaker(row)
        if speaker is None:
            missing_speakers += 1
            continue
        rid = kazemo_filename_stem(row)
        rows.append(
            NormalizedRow(
                dataset=DATASET_KAZEMO,
                row_id=str(rid),
                source_index=idx,
                speaker_id=speaker,
                emotion=_str_or_none(row.get("emotion")),
                gender=None,
                age_category=None,
                augmented=None,
            )
        )
    if missing_speakers:
        warnings.warn(
            f"kazemo: skipped {missing_speakers} rows where speaker could not be parsed",
            stacklevel=2,
        )
    print(f"[sources] kazemo: kept {len(rows)} rows; speakers={sorted({r['speaker_id'] for r in rows})}")
    return rows


DISPATCH = {
    DATASET_BATCH01: load_batch01,
    DATASET_BATCH02: load_batch02,
    DATASET_KAZEMO: load_kazemo,
}


def load_corpus(name: str, cfg: dict[str, Any]) -> list[NormalizedRow]:
    if name not in DISPATCH:
        raise ValueError(f"Unknown corpus {name!r}; expected one of {list(DISPATCH)}")
    return DISPATCH[name](cfg)
