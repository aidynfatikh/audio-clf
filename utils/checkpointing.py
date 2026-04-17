"""Checkpoint save/rotate helpers, cosine LR schedule, and W&B artifact upload."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from utils.misc import REPO_ROOT, unwrap

try:
    import wandb
except Exception:
    wandb = None

# MODEL_DIR is where all per-run outputs land (checkpoints, metrics,
# label encoders, step/latest/best). Override via env var to keep
# experiments separate: `MODEL_DIR=results/exp_batch01 python multihead/train.py`.
_MODEL_DIR_ENV = os.environ.get("MODEL_DIR", "").strip()
MODEL_DIR = Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else REPO_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_wandb_file_artifact(run, *, file_path: Path, name: str, artifact_type: str) -> None:
    if run is None or wandb is None:
        return
    if not file_path.exists():
        return
    art = wandb.Artifact(name=name, type=artifact_type)
    art.add_file(str(file_path), name=file_path.name)
    run.log_artifact(art)


def save_step_checkpoint(
    *,
    step_dir: Path,
    global_step: int,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    best_val_loss: float,
    num_emotions: int,
    num_genders: int,
    num_ages: int,
    samples_seen: int,
) -> Path:
    step_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = step_dir / f"checkpoint_step_{global_step:08d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "samples_seen": samples_seen,
            "source": "step",
            "model_state_dict": unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "num_emotions": num_emotions,
            "num_genders": num_genders,
            "num_ages": num_ages,
        },
        ckpt_path,
    )
    return ckpt_path


def rotate_step_checkpoints(step_dir: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    files = sorted(step_dir.glob("checkpoint_step_*.pt"))
    for p in files[:-keep_last_n]:
        try:
            p.unlink()
        except OSError:
            pass


def make_cosine_schedule(
    optimizer: optim.Optimizer,
    hold_epochs: int,
    decay_epochs: int,
    eta_min_factor: float = 1e-2,
    scale_group_decay: bool = False,
    group_power_range: tuple[float, float] = (0.8, 1.2),
) -> torch.optim.lr_scheduler.LambdaLR:
    """Hold LR for *hold_epochs*, then cosine-anneal to eta_min_factor * base_lr."""

    def base_cosine(epoch: int) -> float:
        if epoch < hold_epochs:
            return 1.0
        t = min((epoch - hold_epochs) / max(decay_epochs - 1, 1), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    if not scale_group_decay:
        def lr_lambda(epoch: int) -> float:
            c = base_cosine(epoch)
            return eta_min_factor + (1.0 - eta_min_factor) * c
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]
    lo, hi = min(lrs), max(lrs)
    p_lo, p_hi = group_power_range
    if hi <= 0 or abs(hi - lo) < 1e-12:
        powers = [1.0 for _ in lrs]
    else:
        powers = [p_lo + (p_hi - p_lo) * ((lr - lo) / (hi - lo)) for lr in lrs]

    lambdas = []
    for p in powers:
        def _f(epoch: int, _p=p) -> float:
            c = base_cosine(epoch)
            return eta_min_factor + (1.0 - eta_min_factor) * (c ** _p)
        lambdas.append(_f)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
