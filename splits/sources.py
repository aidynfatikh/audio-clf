"""Corpus loaders that produce NormalizedRow lists.

No audio bytes are kept here — only the metadata needed to compute splits and
later join back to the source HF datasets via (dataset, row_id).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable

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
    group_fn: Callable[[dict[str, Any], int], str | None] | None = None,
) -> tuple[list[NormalizedRow], dict[str, int]]:
    """Load a HF corpus → list[NormalizedRow].

    `drop_augmented=True` removes aug rows at load time (equivalent to the old
    `filter_augmented: true` / new `augmented_policy: drop_all`). For any other
    policy we return every row with the `augmented` flag populated; the
    builder decides per-split routing.

    `group_fn`, if given, computes the grouping id for each row (used as
    ``speaker_id``). This lets corpora without real speaker labels supply a
    surrogate group — e.g. chunk-family for aug/clean pairs — so related rows
    stay in the same split. When set, it overrides ``speaker_column``.
    """
    # Don't decode audio metadata if we don't need paths; the group_fn may.
    split = _load_hf_metadata_split(hf_id, hf_split, cache_dir)
    columns = set(split.column_names)
    has_augmented_col = "augmented" in columns or "is_augmented" in columns or "metadata" in columns
    has_speaker = group_fn is None and speaker_column is not None and speaker_column in columns

    # When a group_fn is supplied we need access to the audio metadata (paths,
    # filenames) — keep the (undecoded) audio column. Otherwise drop it for speed.
    drop_audio = [c for c in ("audio",) if c in columns and group_fn is None]
    meta_split = split.remove_columns(drop_audio) if drop_audio else split

    rows: list[NormalizedRow] = []
    aug_true = 0
    aug_false = 0
    dropped_aug = 0
    group_fallback = 0
    seen = 0
    for idx, row in enumerate(tqdm(meta_split, desc=f"scan {dataset_name}", leave=False)):
        seen += 1
        aug = extract_augmented_flag(row) if has_augmented_col else None
        if aug is True:
            aug_true += 1
        elif aug is False:
            aug_false += 1
        if drop_augmented and aug is True:
            dropped_aug += 1
            continue
        if group_fn is not None:
            gid = group_fn(row, idx)
            if gid is None:
                group_fallback += 1
            group_val = _str_or_none(gid)
        else:
            group_val = _str_or_none(row.get(speaker_column)) if has_speaker else None
        rows.append(
            NormalizedRow(
                dataset=dataset_name,
                row_id=_row_id_for_hf(row, idx),
                source_index=idx,
                speaker_id=group_val,
                emotion=_str_or_none(row.get("emotion")),
                gender=_str_or_none(row.get("gender")),
                age_category=_str_or_none(row.get("age_category")),
                augmented=aug,
            )
        )
    extra = f", group_fn=on, group_fallback={group_fallback}" if group_fn is not None else ""
    print(
        f"[sources] {dataset_name}: kept {len(rows)} rows "
        f"(aug_true={aug_true}, aug_false={aug_false}, dropped_at_source={dropped_aug}); "
        f"speaker_column_present={has_speaker}, augmented_col_present={has_augmented_col}{extra}"
    )
    stats = {
        "source_rows": seen,
        "kept": len(rows),
        "aug_true": aug_true,
        "aug_false": aug_false,
        "dropped_at_source_aug": dropped_aug,
        "group_fallback": group_fallback,
    }
    return rows, stats


def _batch02_source_recording(row: dict[str, Any], idx: int) -> str | None:
    """Group id that keeps every chunk of the same source recording together.

    ``original_name`` labels the underlying audio source (episode / file) and
    is present on both clean and augmented rows. Grouping by it guarantees
    that all chunks derived from the same physical recording — and their
    augmented copies — land in the same split. This is strictly stronger
    than :func:`_batch02_chunk_family` (per-chunk grouping): that version
    still let chunks from the same source leak across splits.
    """
    name = row.get("original_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    # Fallback: parse the slug from the chunk id / audio path.
    src = row.get("source_chunk_id")
    if not (isinstance(src, str) and src.strip()):
        audio = row.get("audio")
        path = audio.get("path") if isinstance(audio, dict) else None
        src = Path(path).stem if isinstance(path, str) and path else None
        if isinstance(src, str) and src.startswith("aug_"):
            src = src[len("aug_"):]
    if isinstance(src, str) and src.startswith("chunk_"):
        core = src[len("chunk_"):]
        parts = core.rsplit("_", 3)
        if len(parts) == 4:  # slug + seq + start + end
            return parts[0]
        return core
    return None


def _batch02_chunk_family(row: dict[str, Any], idx: int) -> str | None:
    """Group id that keeps every aug copy with its clean parent.

    batch02 rows follow two patterns:
      - clean: audio path = ``chunk_<slug>_<seq>_<start>_<end>.wav`` and
               ``source_chunk_id`` is empty — the row *is* the chunk.
      - aug:   audio path = ``aug_chunk_<slug>..._<hash>.wav`` and
               ``source_chunk_id = "chunk_<slug>_<seq>_<start>_<end>"``
               — pointing at the clean parent.

    Returning the same id for both makes the split builder place every
    aug/clean pair into the same split, so an augmented copy of a held-out
    utterance cannot leak into train.
    """
    src = row.get("source_chunk_id")
    if isinstance(src, str) and src.strip():
        return src.strip()
    audio = row.get("audio")
    path = audio.get("path") if isinstance(audio, dict) else None
    if isinstance(path, str) and path:
        stem = Path(path).stem
        # Clean stems start with "chunk_" and are themselves the chunk id.
        if stem.startswith("chunk_"):
            return stem
        # Aug stems carry a trailing hash: "aug_chunk_<family>_<hash>".
        if stem.startswith("aug_chunk_"):
            core = stem[len("aug_"):]
            return core.rsplit("_", 1)[0] if "_" in core else core
        return stem
    return None


def _should_drop_at_source(cfg: dict[str, Any]) -> bool:
    """Only the `drop_all` policy filters at load time; everything else keeps
    augmented rows so the builder can route them per split.
    """
    policy = _resolve_aug_policy(cfg)
    return policy == "drop_all"


def _resolve_aug_policy(cfg: dict[str, Any]) -> str:
    """Read `augmented_policy` with back-compat for legacy `filter_augmented`.

    Default when neither key is set:
      * ``train_only`` when a ``speaker_column`` is configured — we can attribute
        aug rows to a speaker and prove they don't leak into val/test.
      * ``drop_all`` otherwise — without speaker attribution we can't guarantee
        an aug copy of a held-out utterance won't slip into train.
    """
    raw = cfg.get("augmented_policy")
    if raw:
        p = str(raw).strip().lower()
        if p not in {"drop_all", "keep_all", "train_only", "train_val_only"}:
            raise ValueError(
                "augmented_policy must be drop_all | keep_all | train_only | train_val_only, "
                f"got {raw!r}"
            )
        return p
    if "filter_augmented" in cfg:
        return "drop_all" if bool(cfg["filter_augmented"]) else "keep_all"
    if str(cfg.get("group_by", "")).strip().lower() in {"chunk_family", "source_recording"}:
        return "train_only"
    return "train_only" if cfg.get("speaker_column") else "drop_all"


def load_batch01(cfg: dict[str, Any]) -> tuple[list[NormalizedRow], dict[str, int]]:
    return _load_hf_corpus(
        DATASET_BATCH01,
        hf_id=cfg["hf_id"],
        hf_split=cfg.get("hf_split", "train"),
        cache_dir=Path(cfg["cache_dir"]),
        speaker_column=cfg.get("speaker_column", "speaker_id"),
        drop_augmented=_should_drop_at_source(cfg),
    )


def load_batch02(cfg: dict[str, Any]) -> tuple[list[NormalizedRow], dict[str, int]]:
    # batch02 has no speaker column. We derive a chunk-family group id from
    # `source_chunk_id` (aug rows) or the audio-path stem (clean rows). The
    # key pairs each aug copy with its clean parent so they land in the same
    # split — prevents aug→train / clean→val leakage without any speaker labels.
    # Callers can opt out via `group_by: none`.
    group_by = str(cfg.get("group_by", "source_recording")).strip().lower()
    group_fn: Callable[[dict[str, Any], int], str | None] | None
    if group_by == "source_recording":
        group_fn = _batch02_source_recording
    elif group_by == "chunk_family":
        group_fn = _batch02_chunk_family
    elif group_by in ("", "none", "off"):
        group_fn = None
    else:
        raise ValueError(f"batch02 group_by must be source_recording | chunk_family | none, got {group_by!r}")
    return _load_hf_corpus(
        DATASET_BATCH02,
        hf_id=cfg["hf_id"],
        hf_split=cfg.get("hf_split", "train"),
        cache_dir=Path(cfg["cache_dir"]),
        speaker_column=cfg.get("speaker_column", "speaker_id"),
        drop_augmented=_should_drop_at_source(cfg),
        group_fn=group_fn,
    )


def load_kazemo(cfg: dict[str, Any]) -> tuple[list[NormalizedRow], dict[str, int]]:
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
    stats = {
        "source_rows": len(rows) + missing_speakers,
        "kept": len(rows),
        "dropped_missing_speaker": missing_speakers,
    }
    return rows, stats


DISPATCH = {
    DATASET_BATCH01: load_batch01,
    DATASET_BATCH02: load_batch02,
    DATASET_KAZEMO: load_kazemo,
}


def load_corpus(name: str, cfg: dict[str, Any]) -> tuple[list[NormalizedRow], dict[str, int]]:
    if name not in DISPATCH:
        raise ValueError(f"Unknown corpus {name!r}; expected one of {list(DISPATCH)}")
    return DISPATCH[name](cfg)
