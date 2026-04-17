"""Parse the speaker token from KazEmoTTS rows.

KazEmoTTS encodes identity in filenames/text as `<speaker>_<emotion>_<utt>`,
e.g. `aqtolkyn_happy_0102.wav`. We reuse the emotion vocabulary from
loaders.kazemo.load_data (via _EMO_MAP) to locate the emotion token, then
take everything before it as the speaker name.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any

from loaders.kazemo.load_data import _EMO_MAP

from splits.schema import (
    NormalizedRow,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)


def _candidates(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    t = row.get("text")
    if isinstance(t, str) and t:
        out.append(t.split("|", 1)[0])
    a = row.get("audio")
    if isinstance(a, dict):
        p = a.get("path")
        if isinstance(p, str) and p:
            out.append(Path(p).name)
    rid = row.get("row_id") or row.get("id")
    if isinstance(rid, str) and rid:
        out.append(rid)
    return out


def parse_kazemo_speaker(row: dict[str, Any]) -> str | None:
    """Return normalized speaker id (lowercase), or None if parse fails."""
    for raw in _candidates(row):
        stem = Path(raw).stem
        tokens = re.split(r"[_\W]+", stem)
        if not tokens:
            continue
        speaker_parts: list[str] = []
        for tok in tokens:
            if not tok:
                continue
            if tok.lower() in _EMO_MAP:
                # emotion token reached — everything before it is the speaker
                if speaker_parts:
                    return "_".join(speaker_parts).lower()
                return None
            speaker_parts.append(tok)
    return None


def kazemo_filename_stem(row: dict[str, Any]) -> str:
    """Stable row id for a kazemo row (filename stem, fallback to text prefix)."""
    a = row.get("audio")
    if isinstance(a, dict):
        p = a.get("path")
        if isinstance(p, str) and p:
            return Path(p).stem
    t = row.get("text")
    if isinstance(t, str) and t:
        prefix = t.split("|", 1)[0]
        return Path(prefix).stem
    # Last resort — unstable across scans; callers should have a better id.
    return f"kazemo_row_{id(row)}"


def kazemo_three_way_split(
    rows: list[NormalizedRow],
    train_speakers: set[str],
    valtest_speaker: str,
    valtest_ratio: dict[str, float],
    seed: int,
) -> dict[str, list[int]]:
    """Deterministic 3-way split for kazemo.

    Rows from speakers in ``train_speakers`` go to train.
    Rows from ``valtest_speaker`` are randomly (seeded) partitioned between
    val and test according to ``valtest_ratio``.
    Rows from any other speaker are dropped with a clear message.
    """
    val_frac = float(valtest_ratio.get(SPLIT_VAL, 0.5))
    test_frac = float(valtest_ratio.get(SPLIT_TEST, 1.0 - val_frac))
    if val_frac <= 0 or test_frac <= 0:
        raise ValueError(f"valtest_ratio must have positive val/test, got {valtest_ratio}")
    # Normalize
    total = val_frac + test_frac
    val_frac /= total

    out: dict[str, list[int]] = {SPLIT_TRAIN: [], SPLIT_VAL: [], SPLIT_TEST: []}
    valtest_indices: list[int] = []
    for i, r in enumerate(rows):
        spk = r.get("speaker_id")
        if spk is None:
            continue
        if spk in train_speakers:
            out[SPLIT_TRAIN].append(i)
        elif spk == valtest_speaker:
            valtest_indices.append(i)
        # else: speaker not assigned — skipped

    rng = random.Random(seed)
    shuffled = list(valtest_indices)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_frac))
    out[SPLIT_VAL] = sorted(shuffled[:n_val])
    out[SPLIT_TEST] = sorted(shuffled[n_val:])
    return out


def resolve_kazemo_speakers(
    all_speakers: list[str],
    *,
    strategy: str,
    train_indices: list[int] | None = None,
    valtest_index: int | None = None,
    train_names: list[str] | None = None,
    valtest_name: str | None = None,
) -> tuple[set[str], str]:
    """Resolve config-specified speakers against the actually-scanned set.

    Returns (train_speakers_set, valtest_speaker).
    """
    sorted_speakers = sorted(all_speakers)
    if strategy == "by_index":
        if train_indices is None or valtest_index is None:
            raise ValueError("by_index strategy requires train_indices and valtest_index")
        try:
            train = {sorted_speakers[i] for i in train_indices}
            valtest = sorted_speakers[valtest_index]
        except IndexError as e:
            raise ValueError(
                f"kazemo speaker indices out of range; discovered speakers={sorted_speakers}"
            ) from e
    elif strategy == "by_name":
        if not train_names or not valtest_name:
            raise ValueError("by_name strategy requires train_names and valtest_name")
        missing = [n for n in (*train_names, valtest_name) if n not in sorted_speakers]
        if missing:
            raise ValueError(
                f"kazemo speakers not found in scan: {missing}; discovered={sorted_speakers}"
            )
        train = set(train_names)
        valtest = valtest_name
    else:
        raise ValueError(f"Unknown kazemo speaker_selection.strategy: {strategy!r}")

    if valtest in train:
        raise ValueError(f"kazemo valtest speaker {valtest!r} overlaps train set {sorted(train)}")
    return train, valtest
