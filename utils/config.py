"""Stage config loader for multihead/train.py and multihead/finetune.py.

Reads `configs/train/hubert_stage{N}.yaml` and layers env overrides on top for
the runtime-only knobs that legitimately vary per invocation (W&B credentials,
BATCH_SIZE, NOISE_DIR). Everything else lives in YAML.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

from utils.misc import REPO_ROOT, resolve_batch_size

_CONFIG_DIR = REPO_ROOT / "configs" / "train"


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes"}


def load_stage_config(stage: int) -> dict[str, Any]:
    """Load stage YAML and attach a ``runtime`` block derived from env vars.

    Returns the parsed YAML dict with a ``runtime`` key added containing:
      - batch_size: resolved from BATCH_SIZE env var, else default_batch_size
      - noise_dir: NOISE_DIR env var (or None)
      - wandb: dict with all WANDB_* overrides
      - checkpointing overrides: CHECKPOINT_EVERY_STEPS, VAL_EVERY_STEPS, etc.
    """
    path = _CONFIG_DIR / f"hubert_stage{stage}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Stage config not found: {path}")
    with open(path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    default_bs = int(cfg["training"].get("default_batch_size", 4))
    ckpt = cfg.get("checkpointing", {})
    val_cfg = cfg.get("validation", {})

    cfg["runtime"] = {
        "batch_size": resolve_batch_size(default=default_bs),
        "noise_dir": os.environ.get("NOISE_DIR"),
        "checkpoint_every_steps": int(os.environ.get("CHECKPOINT_EVERY_STEPS", "0")),
        "num_checkpoints": int(os.environ.get("NUM_CHECKPOINTS", str(ckpt.get("num_checkpoints", 5)))),
        "keep_last_n": int(os.environ.get("CHECKPOINT_KEEP_LAST_N_STEP_FILES",
                                          str(ckpt.get("keep_last_n", 5)))),
        "save_latest_every_steps": _truthy(os.environ.get(
            "CHECKPOINT_SAVE_LATEST_EVERY_STEPS",
            "1" if ckpt.get("save_latest_every_steps") else "0",
        )),
        "num_validations": int(os.environ.get("NUM_VALIDATIONS", str(val_cfg.get("num_validations", 5)))),
        "val_every_steps": int(os.environ.get("VAL_EVERY_STEPS", "0")),
        "wandb": {
            "enabled": _truthy(os.environ.get("WANDB_ENABLED", "0")),
            "entity": os.environ.get("WANDB_ENTITY", ""),
            "project": os.environ.get("WANDB_PROJECT", "audio-clf"),
            "run_name": os.environ.get("WANDB_RUN_NAME", ""),
            "run_group": os.environ.get("WANDB_RUN_GROUP", ""),
            "mode": os.environ.get("WANDB_MODE", ""),
            "tags": [t.strip() for t in os.environ.get("WANDB_TAGS", "").split(",") if t.strip()],
            "upload_best": _truthy(os.environ.get("WANDB_UPLOAD_BEST_ARTIFACT", "1")),
            "upload_latest": _truthy(os.environ.get("WANDB_UPLOAD_LATEST_ARTIFACT", "1")),
            "upload_step": _truthy(os.environ.get("WANDB_UPLOAD_STEP_ARTIFACT", "1")),
            "latest_every_steps": int(os.environ.get("WANDB_LATEST_ARTIFACT_EVERY_STEPS", "0")),
        },
    }
    return cfg
