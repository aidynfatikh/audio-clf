from __future__ import annotations

import os
import re
from typing import Any

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["TORCHCODEC_QUIET"] = "1"

from datasets import Audio, DatasetDict, load_dataset


DATASET_ID = "issai/KazEmoTTS"

_EMO_MAP = {
    "scared": "fearful",
    "fear": "fearful",
    "fearful": "fearful",
    "surprise": "surprised",
    "surprised": "surprised",
    "neutral": "neutral",
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
}


def _extract_emotion(row: dict[str, Any]) -> str | None:
    """
    KazEmoTTS rows often encode emotion inside `text` like:
      "<speaker>_<emotion>_<utt>|<text>"
    We also fall back to the audio path filename when possible.
    """
    candidates: list[str] = []

    t = row.get("text")
    if isinstance(t, str) and t:
        candidates.append(t.split("|", 1)[0])

    a = row.get("audio")
    if isinstance(a, dict):
        p = a.get("path")
        if isinstance(p, str) and p:
            candidates.append(os.path.basename(p))

    for s in candidates:
        parts = re.split(r"[_\W]+", s.lower())
        for tok in parts:
            if tok in _EMO_MAP:
                return _EMO_MAP[tok]
    return None


def load_kazemotts(cache_dir: str | None = None) -> DatasetDict:
    # KazEmoTTS is sometimes packaged in a way that triggers UnicodeDecodeError
    # in the default "text" loader path. Retry with more permissive decoding.
    attempts = [
        {},
        {"errors": "replace"},
        {"errors": "ignore"},
        {"encoding": "utf-8", "errors": "replace"},
        {"encoding": "latin-1"},
        {"encoding": "cp1251", "errors": "replace"},
    ]

    last_err: Exception | None = None
    ds = None
    for extra in attempts:
        try:
            ds = load_dataset(DATASET_ID, cache_dir=cache_dir, **extra)
            last_err = None
            break
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except TypeError as e:
            # Some dataset builders don't accept encoding/errors kwargs.
            last_err = e
            continue

    if ds is None:
        raise RuntimeError(
            "Failed to load issai/KazEmoTTS due to a text decoding/build issue. "
            "Try setting a custom cache dir and re-running, or re-upload the dataset "
            "with UTF-8 metadata. Last error: "
            + repr(last_err)
        )

    def _add_emotion(example: dict[str, Any]) -> dict[str, Any]:
        emo = _extract_emotion(example)
        if emo is None:
            raise ValueError("Could not extract emotion from row (missing/unknown format).")
        example["emotion"] = emo
        return example

    out = DatasetDict()
    for split_name, split in ds.items():
        if "audio" in split.column_names:
            split = split.cast_column("audio", Audio(decode=False))
        split = split.map(_add_emotion)
        out[split_name] = split
    return out

