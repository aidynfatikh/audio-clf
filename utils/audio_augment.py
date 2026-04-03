from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np


def speed_perturb(wav: np.ndarray, factor: float) -> np.ndarray:
    """
    Change speed by resampling in the time domain (keeps sample rate metadata unchanged).
    factor > 1.0 => faster (shorter); factor < 1.0 => slower (longer).
    """
    if factor == 1.0:
        return wav
    n = int(round(len(wav) / float(factor)))
    n = max(n, 1)
    x_old = np.arange(len(wav), dtype=np.float32)
    x_new = np.linspace(0.0, float(len(wav) - 1), num=n, dtype=np.float32)
    return np.interp(x_new, x_old, wav).astype(np.float32, copy=False)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Mix noise into clean at target SNR (dB), returning clean + scaled_noise.
    """
    clean_rms = _rms(clean)
    noise_rms = _rms(noise)
    if noise_rms <= 0.0:
        return clean
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scale = target_noise_rms / noise_rms
    return (clean + noise * scale).astype(np.float32, copy=False)


def _list_audio_files(root: Path) -> list[Path]:
    exts = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def _crop_or_tile(x: np.ndarray, length: int) -> np.ndarray:
    if len(x) >= length:
        start = random.randint(0, len(x) - length) if len(x) > length else 0
        return x[start : start + length]
    # tile to length
    reps = int(np.ceil(length / max(len(x), 1)))
    tiled = np.tile(x, reps)[:length]
    return tiled.astype(np.float32, copy=False)


@dataclass
class NoiseMixer:
    noise_paths: list[Path]

    @classmethod
    def from_dir(cls, noise_dir: str | Path) -> "NoiseMixer | None":
        p = Path(noise_dir).expanduser()
        if not p.exists() or not p.is_dir():
            return None
        files = _list_audio_files(p)
        if not files:
            return None
        return cls(files)

    def sample(self, length: int) -> np.ndarray | None:
        if not self.noise_paths:
            return None
        path = random.choice(self.noise_paths)
        try:
            import soundfile as sf

            noise, sr = sf.read(str(path), always_2d=False)
        except Exception:
            return None
        noise = np.asarray(noise, dtype=np.float32)
        if noise.ndim > 1:
            noise = noise.mean(axis=1).astype(np.float32, copy=False)
        return _crop_or_tile(noise, length)
