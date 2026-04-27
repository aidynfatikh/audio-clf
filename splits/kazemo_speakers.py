"""Parse the speaker token from KazEmoTTS rows.

KazEmoTTS encodes identity in filenames/text as `<speaker>_<emotion>_<utt>`,
e.g. `aqtolkyn_happy_0102.wav`. We reuse the emotion vocabulary from
loaders.kazemo.load_data (via _EMO_MAP) to locate the emotion token, then
take everything before it as the speaker name.
"""

from __future__ import annotations

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
            # Keep full path so parent-dir narrator tokens (e.g. .../F1/x.wav)
            # remain visible to parse_kazemo_speaker.
            out.append(p)
    rid = row.get("row_id") or row.get("id")
    if isinstance(rid, str) and rid:
        out.append(rid)
    return out


_NARRATOR_RE = re.compile(r"^[fm]\d+$", re.IGNORECASE)


def parse_kazemo_speaker(row: dict[str, Any]) -> str | None:
    """Return normalized speaker id (lowercase), or None if parse fails.

    KazEmoTTS has 3 narrators (F1, M1, M2). Filenames in the EmoKaz.zip use
    a `<utt_id>_<emotion>_<narrator>.wav` layout — utterance-id-first, not
    speaker-first. We first scan every token (across all candidate strings:
    text field, audio path, parent dirs) for a narrator-like token (F\\d+ /
    M\\d+); if one is found, that's the speaker. If no narrator token is
    present anywhere, fall back to the legacy "everything before the first
    emotion token" heuristic for compatibility with non-EmoKaz layouts.
    """
    for raw in _candidates(row):
        # Walk full path so parent directories like "F1/" also count.
        p = Path(raw)
        path_tokens: list[str] = []
        for part in (*p.parts[:-1], p.stem):
            path_tokens.extend(t for t in re.split(r"[_\W]+", part) if t)
        for tok in path_tokens:
            if _NARRATOR_RE.match(tok):
                return tok.lower()

    # Fallback: legacy <speaker>_<emotion>_<utt> layout.
    for raw in _candidates(row):
        stem = Path(raw).stem
        tokens = [t for t in re.split(r"[_\W]+", stem) if t]
        speaker_parts: list[str] = []
        for tok in tokens:
            if tok.lower() in _EMO_MAP:
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
    valtest_speakers: set[str],
    valtest_ratio: dict[str, float],
    seed: int,
) -> dict[str, list[int]]:
    """Deterministic 3-way split for kazemo.

    Rows from speakers in ``train_speakers`` go to train.
    Rows from speakers in ``valtest_speakers`` are randomly (seeded)
    partitioned between val and test according to ``valtest_ratio``.
    A speaker appearing in both sets raises — speaker-disjoint is required.
    Rows from speakers in neither set are dropped.
    """
    overlap = train_speakers & valtest_speakers
    if overlap:
        raise ValueError(f"kazemo: speaker(s) {sorted(overlap)} appear in both train and valtest sets")

    val_frac = float(valtest_ratio.get(SPLIT_VAL, 0.5))
    test_frac = float(valtest_ratio.get(SPLIT_TEST, 1.0 - val_frac))
    if val_frac <= 0 or test_frac <= 0:
        raise ValueError(f"valtest_ratio must have positive val/test, got {valtest_ratio}")
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
        elif spk in valtest_speakers:
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
) -> tuple[set[str], set[str]]:
    """Resolve config-specified speakers against the actually-scanned set.

    Returns (train_speakers_set, valtest_speakers_set). The ``valtest_all``
    strategy puts every discovered speaker into val/test with an empty train
    set — used when kazemo is held out as an evaluation-only corpus.
    """
    sorted_speakers = sorted(all_speakers)
    if strategy == "valtest_all":
        return set(), set(sorted_speakers)
    if strategy == "by_index":
        if train_indices is None or valtest_index is None:
            raise ValueError("by_index strategy requires train_indices and valtest_index")
        try:
            train = {sorted_speakers[i] for i in train_indices}
            valtest = {sorted_speakers[valtest_index]}
        except IndexError as e:
            raise ValueError(
                f"kazemo speaker indices out of range; discovered speakers={sorted_speakers}"
            ) from e
    elif strategy == "by_name":
        if not train_names or not valtest_name:
            raise ValueError("by_name strategy requires train_names and valtest_name")
        _valtest_names = valtest_name if isinstance(valtest_name, list) else [valtest_name]
        missing = [n for n in (*train_names, *_valtest_names) if n not in sorted_speakers]
        if missing:
            raise ValueError(
                f"kazemo speakers not found in scan: {missing}; discovered={sorted_speakers}"
            )
        train = set(train_names)
        valtest = set(_valtest_names)
    else:
        raise ValueError(f"Unknown kazemo speaker_selection.strategy: {strategy!r}")

    overlap = valtest & train
    if overlap:
        raise ValueError(f"kazemo valtest speaker(s) {sorted(overlap)} overlap train set {sorted(train)}")
    return train, valtest
