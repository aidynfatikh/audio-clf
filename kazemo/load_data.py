from __future__ import annotations

import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["TORCHCODEC_QUIET"] = "1"

from datasets import Audio, DatasetDict, load_dataset
from datasets.exceptions import DatasetGenerationError


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


def _log(msg: str) -> None:
    print(msg, flush=True)


def _extract_emotion_from_name(name: str) -> str | None:
    """Extract normalized emotion token from an archive/file name."""
    parts = re.split(r"[_\W]+", Path(name).name.lower())
    for tok in parts:
        if tok in _EMO_MAP:
            return _EMO_MAP[tok]
    return None


def _select_balanced_members(members: list[str], max_samples: int | None) -> list[tuple[str, str]]:
    """
    Return [(member_name, emotion)] with balanced sampling across emotions.

    Sampling is deterministic for reproducibility:
      - files are grouped by emotion using filename parsing
      - each group sorted lexicographically
      - round-robin across emotions until max_samples (or groups exhausted)
    """
    by_emotion: dict[str, list[str]] = {}
    for member in members:
        emo = _extract_emotion_from_name(member)
        if emo is None:
            continue
        by_emotion.setdefault(emo, []).append(member)

    if not by_emotion:
        return []

    for emo in by_emotion:
        by_emotion[emo].sort()

    emotions = sorted(by_emotion.keys())
    if max_samples is None or max_samples <= 0:
        max_samples = sum(len(v) for v in by_emotion.values())

    selected: list[tuple[str, str]] = []
    idx = {emo: 0 for emo in emotions}
    while len(selected) < max_samples:
        progressed = False
        for emo in emotions:
            i = idx[emo]
            pool = by_emotion[emo]
            if i < len(pool):
                selected.append((pool[i], emo))
                idx[emo] = i + 1
                progressed = True
                if len(selected) >= max_samples:
                    break
        if not progressed:
            break
    return selected


def _load_from_zip_fallback(cache_dir: str | None, max_samples: int | None) -> DatasetDict:
    """
    Fallback loader for when HF `datasets` mistakenly routes this repo through the
    generic `text` builder and tries to UTF-8 decode a binary zip (e.g. EmoKaz.zip).

    Strategy:
      - snapshot_download the dataset repo
      - find the biggest .zip file
      - select a balanced subset by emotion from archive members
      - extract only selected files into cache_dir
      - build a Dataset from selected files
    """
    from huggingface_hub import snapshot_download
    from datasets import Dataset

    _log("Loading KazEmoTTS from zip (skip load_dataset)...")
    base_cache = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "huggingface")
    repo_dir = base_cache / "kazemo" / "repo"
    extracted_dir = base_cache / "kazemo" / "extracted_partial"

    repo_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=str(repo_dir),
        local_dir_use_symlinks=False,
    )
    _log("Repo ready. Locating zip...")

    zips = sorted(repo_dir.rglob("*.zip"), key=lambda p: p.stat().st_size, reverse=True)
    if not zips:
        raise FileNotFoundError(f"No .zip found in downloaded dataset repo at {repo_dir}")

    zip_path = zips[0]
    _log(f"Found {zip_path.name} ({zip_path.stat().st_size / 1e9:.1f} GB).")

    audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    _log("Indexing zip members and selecting balanced subset...")
    with zipfile.ZipFile(zip_path) as zf:
        audio_members = [
            name for name in zf.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() in audio_exts
        ]
        if not audio_members:
            raise RuntimeError(f"Zip {zip_path} contains no audio members.")

        selected = _select_balanced_members(audio_members, max_samples=max_samples)
        if not selected:
            raise RuntimeError("Could not infer emotion labels from zip member names.")

        _log(f"Selected {len(selected)} files (balanced by emotion).")
        selected_paths: list[str] = []
        selected_emotions: list[str] = []
        for member_name, emo in selected:
            target_path = extracted_dir / member_name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.exists() or target_path.stat().st_size == 0:
                with zf.open(member_name) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            selected_paths.append(str(target_path))
            selected_emotions.append(emo)

    _log(f"Prepared {len(selected_paths)} extracted audio files.")
    ds = Dataset.from_dict({"audio": selected_paths, "emotion": selected_emotions})
    ds = ds.cast_column("audio", Audio(decode=False))
    return DatasetDict({"train": ds})


def load_kazemotts(cache_dir: str | None = None, max_samples: int | None = None) -> DatasetDict:
    # The HF hub repo currently points to a binary `.zip` but is sometimes loaded
    # with the generic `text` builder, which tries to UTF-8 decode that zip.
    # We try the default loader and one permissive config; if it still fails,
    # we fall back to downloading/extracting the zip and enumerating audio files.
    #
    # Important: if max_samples is set, we skip the `load_dataset` path entirely
    # to avoid building a huge (and sometimes broken) intermediate "text" split,
    # while enforcing balanced per-emotion sampling via zip fallback.
    if max_samples is not None and max_samples > 0:
        ds = _load_from_zip_fallback(cache_dir, max_samples)
    else:
        try:
            ds = load_dataset(DATASET_ID, cache_dir=cache_dir)
        except (UnicodeDecodeError, DatasetGenerationError, TypeError, ValueError):
            try:
                ds = load_dataset(
                    DATASET_ID,
                    cache_dir=cache_dir,
                    encoding="utf-8",
                    encoding_errors="replace",
                )
            except (UnicodeDecodeError, DatasetGenerationError, TypeError, ValueError):
                ds = _load_from_zip_fallback(cache_dir, max_samples)

    def _add_emotion(example: dict[str, Any]) -> dict[str, Any]:
        if isinstance(example.get("emotion"), str) and example.get("emotion"):
            return example
        emo = _extract_emotion(example)
        if emo is None:
            raise ValueError("Could not extract emotion from row (missing/unknown format).")
        example["emotion"] = emo
        return example

    out = DatasetDict()
    for split_name, split in ds.items():
        _log(f"Preparing split '{split_name}' ({len(split)} rows)...")
        if "audio" in split.column_names:
            split = split.cast_column("audio", Audio(decode=False))
        split = split.map(_add_emotion, desc="Adding emotion labels")
        out[split_name] = split
    return out

