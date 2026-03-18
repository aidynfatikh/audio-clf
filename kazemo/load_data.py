from __future__ import annotations

import os
import re
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


def _load_from_zip_fallback(cache_dir: str | None, max_samples: int | None) -> DatasetDict:
    """
    Fallback loader for when HF `datasets` mistakenly routes this repo through the
    generic `text` builder and tries to UTF-8 decode a binary zip (e.g. EmoKaz.zip).

    Strategy:
      - snapshot_download the dataset repo
      - find the biggest .zip file
      - extract it once into cache_dir
      - build a Dataset from discovered audio files (emotion derived from filename)
    """
    from huggingface_hub import snapshot_download
    from datasets import Dataset

    _log("Loading KazEmoTTS from zip (skip load_dataset)...")
    base_cache = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "huggingface")
    repo_dir = base_cache / "kazemo" / "repo"
    extracted_dir = base_cache / "kazemo" / "extracted"
    extracted_marker = extracted_dir / ".extracted.ok"

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

    if not extracted_marker.exists():
        _log("Extracting zip (this may take several minutes)...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extracted_dir)
        extracted_marker.write_text(str(zip_path), encoding="utf-8")
        _log("Extraction done.")
    else:
        _log("Using already-extracted files.")

    _log("Scanning for audio files...")
    audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    audio_files = [p for p in extracted_dir.rglob("*") if p.suffix.lower() in audio_exts]
    if not audio_files:
        raise RuntimeError(f"Extracted {zip_path} but found no audio files in {extracted_dir}")

    audio_files = sorted(audio_files)
    if max_samples is not None and max_samples > 0:
        audio_files = audio_files[: min(max_samples, len(audio_files))]
    _log(f"Using {len(audio_files)} audio files.")

    ds = Dataset.from_dict({"audio": [str(p) for p in audio_files]})
    ds = ds.cast_column("audio", Audio(decode=False))
    return DatasetDict({"train": ds})


def load_kazemotts(cache_dir: str | None = None, max_samples: int | None = None) -> DatasetDict:
    # The HF hub repo currently points to a binary `.zip` but is sometimes loaded
    # with the generic `text` builder, which tries to UTF-8 decode that zip.
    # We try the default loader and one permissive config; if it still fails,
    # we fall back to downloading/extracting the zip and enumerating audio files.
    #
    # Important: if max_samples is set, we skip the `load_dataset` path entirely
    # to avoid building a huge (and sometimes broken) intermediate "text" split.
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

