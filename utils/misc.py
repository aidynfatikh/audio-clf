"""Constants and small env-agnostic helpers shared across all modules."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_RATE = 16_000  # HuBERT requires 16 kHz
RANDOM_SEED = 42
BATCH_SIZE_ENV_VAR = "BATCH_SIZE"

# Task frozensets — used by both data_loading and training modules.
_ALL_TASKS: frozenset = frozenset({"emotion", "gender", "age"})
_KAZEMO_TASKS: frozenset = frozenset({"emotion"})

# Mutable: set to True by SIGINT handler so training loops can exit cleanly.
stop_requested = False


def resolve_batch_size(default: int) -> int:
    """Return BATCH_SIZE env var as int, or *default* if unset."""
    raw = os.environ.get(BATCH_SIZE_ENV_VAR)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{BATCH_SIZE_ENV_VAR} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{BATCH_SIZE_ENV_VAR} must be > 0, got {value}")
    return value


def apply_cuda_perf_flags(device: torch.device | None = None) -> None:
    """Set TF32 / cuDNN / matmul precision defaults for CUDA training."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set all global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sigint_handler(signum, frame) -> None:
    global stop_requested
    if stop_requested:
        print("\nSecond Ctrl+C: exiting immediately.", file=sys.stderr)
        sys.exit(130)
    stop_requested = True
    print("\nCtrl+C received. Finishing current batch and saving checkpoint...")


def unwrap(model: nn.Module) -> nn.Module:
    """Return the base module, unwrapping torch.compile's OptimizedModule if present."""
    return getattr(model, "_orig_mod", model)
