#!/usr/bin/env python3
"""
Fine-tune MultiTaskHubert by partially unfreezing backbone layers.

Stage 2 training: loads a checkpoint from frozen-backbone training (train.py),
then unfreezes the top-N HuBERT transformer layers (ranked by learned per-task
layer weights), and continues training with a low backbone LR.

Usage:
  python finetune.py              # Fine-tune with defaults (top 4 layers)
  python finetune.py --analyze    # Print layer importance and exit
"""

import os
import signal
import sys
import warnings
from datetime import datetime

os.environ['DATASETS_AUDIO_BACKEND'] = 'soundfile'
os.environ['TORCHCODEC_QUIET'] = '1'
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor
from pathlib import Path
from tqdm import tqdm
import json

try:
    import wandb
except Exception:
    wandb = None

from train import (
    MultiTaskHubert,
    AudioDataset,
    build_label_encoders,
    build_mixed_train_val_splits,
    train_epoch,
    validate,
    _sigint_handler,
    _unwrap,
    _make_cosine_schedule,
    set_seed,
    RANDOM_SEED,
)

# ── Configuration ────────────────────────────────────────────────────────────
BATCH_SIZE = 8
NUM_EPOCHS = 20
SAMPLE_RATE = 16000

# Learning rates for stage-2 fine-tuning
# BACKBONE_LR_TOP is the learning rate for the highest (most task-relevant) encoder layer.
# Lower layers receive a geometrically decayed rate:  LR_n = BACKBONE_LR_TOP * LAYER_DECAY^(12-n)
BACKBONE_LR_TOP = 5e-5   # Top unfrozen encoder layer  (pre-decay baseline)
LAYER_DECAY = 0.85        # ξ in the decay formula; 0.8–0.9 is the typical range
HEAD_LR = 2e-4            # Slightly lower than stage-1 heads (already trained)

# Loss weights (same as stage-1)
EMOTION_WEIGHT = 1.2
GENDER_WEIGHT = 0.5
AGE_WEIGHT = 1.0

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# All fine-tune outputs go here so they never overwrite stage-1 files
FINETUNE_DIR = MODEL_DIR / "finetune"
FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

# Fine-tune defaults (no CLI)
UNFREEZE_TOP_N = 4
UNFREEZE_FEATURE_PROJ = False
ANALYZE_ONLY = "--analyze" in sys.argv
EARLY_STOPPING_PATIENCE = 5

# HuBERT has 13 hidden states: index 0 = feature-projection output,
# indices 1-12 = transformer encoder layers [0..11].
NUM_HUBERT_LAYERS = 13
NUM_TRANSFORMER_LAYERS = 12  # encoder.layers[0] … encoder.layers[11]

_stop_requested = False

CHECKPOINT_EVERY_STEPS = int(os.environ.get("CHECKPOINT_EVERY_STEPS", "0"))
CHECKPOINT_KEEP_LAST_N_STEP_FILES = int(os.environ.get("CHECKPOINT_KEEP_LAST_N_STEP_FILES", "5"))
CHECKPOINT_SAVE_LATEST_EVERY_STEPS = os.environ.get("CHECKPOINT_SAVE_LATEST_EVERY_STEPS", "0").strip().lower() in {"1", "true", "yes"}

WANDB_ENABLED = os.environ.get("WANDB_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "audio-clf")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", "")
WANDB_RUN_GROUP = os.environ.get("WANDB_RUN_GROUP", "")
WANDB_MODE = os.environ.get("WANDB_MODE", "")
WANDB_TAGS = [t.strip() for t in os.environ.get("WANDB_TAGS", "").split(",") if t.strip()]
WANDB_UPLOAD_BEST_ARTIFACT = os.environ.get("WANDB_UPLOAD_BEST_ARTIFACT", "1").strip().lower() in {"1", "true", "yes"}
WANDB_UPLOAD_LATEST_ARTIFACT = os.environ.get("WANDB_UPLOAD_LATEST_ARTIFACT", "1").strip().lower() in {"1", "true", "yes"}
WANDB_UPLOAD_STEP_ARTIFACT = os.environ.get("WANDB_UPLOAD_STEP_ARTIFACT", "1").strip().lower() in {"1", "true", "yes"}
WANDB_LATEST_ARTIFACT_EVERY_STEPS = int(os.environ.get("WANDB_LATEST_ARTIFACT_EVERY_STEPS", "0"))
# ─────────────────────────────────────────────────────────────────────────────


def _save_wandb_file_artifact(run, *, file_path: Path, name: str, artifact_type: str) -> None:
    if run is None or wandb is None:
        return
    if not file_path.exists():
        return
    art = wandb.Artifact(name=name, type=artifact_type)
    art.add_file(str(file_path), name=file_path.name)
    run.log_artifact(art)


def _save_step_checkpoint(
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
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'samples_seen': samples_seen,
        'source': 'step',
        'model_state_dict': _unwrap(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'num_emotions': num_emotions,
        'num_genders': num_genders,
        'num_ages': num_ages,
    }, ckpt_path)
    return ckpt_path


def _rotate_step_checkpoints(step_dir: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    files = sorted(step_dir.glob("checkpoint_step_*.pt"))
    stale = files[:-keep_last_n]
    for p in stale:
        try:
            p.unlink()
        except OSError:
            pass


# ── Layer-importance analysis ─────────────────────────────────────────────────

def load_layer_prefs(metrics_path: Path) -> dict[str, list[float]] | None:
    """Return averaged softmax layer weights from the last epoch of a metrics file."""
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        data = json.load(f)
    if not data:
        return None
    last = data[-1]
    return last.get("layer_prefs")


def aggregate_layer_importance(metrics_dir: Path) -> dict[str, list[float]]:
    """
    Aggregate layer preferences across all training_metrics_*.json files.
    Returns per-task list of length 13 (summed softmax weights, then renormalised).
    """
    tasks = ["emotion", "gender", "age"]
    sums: dict[str, list[float]] = {t: [0.0] * NUM_HUBERT_LAYERS for t in tasks}
    n_files = 0

    candidate_files = list(metrics_dir.glob("training_metrics*.json"))
    for p in candidate_files:
        prefs = load_layer_prefs(p)
        if prefs is None:
            continue
        n_files += 1
        for task in tasks:
            for i, v in enumerate(prefs.get(task, [])):
                sums[task][i] += v

    if n_files == 0:
        # Fallback: uniform
        return {t: [1.0 / NUM_HUBERT_LAYERS] * NUM_HUBERT_LAYERS for t in tasks}

    # Normalise
    result = {}
    for task in tasks:
        total = sum(sums[task]) or 1.0
        result[task] = [v / total for v in sums[task]]
    return result


def latest_layer_prefs(metrics_dir: Path) -> dict[str, list[float]]:
    """
    Return the per-task layer preferences from the **last epoch** of the
    **most recently modified** training_metrics*.json file.

    This is used for the default unfreeze ranking instead of the multi-run
    average, so we act on the most up-to-date network state.
    Falls back to aggregate_layer_importance if no file is found.
    """
    tasks = ["emotion", "gender", "age"]
    candidates = list(metrics_dir.glob("training_metrics*.json"))
    if not candidates:
        return aggregate_layer_importance(metrics_dir)

    # Most recently modified file
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    prefs = load_layer_prefs(newest)
    if prefs is None:
        return aggregate_layer_importance(metrics_dir)

    # Normalise each task so values sum to 1
    result = {}
    for task in tasks:
        raw = prefs.get(task, [1.0 / NUM_HUBERT_LAYERS] * NUM_HUBERT_LAYERS)
        total = sum(raw) or 1.0
        result[task] = [v / total for v in raw]
    return result


def rank_transformer_layers(layer_prefs: dict[str, list[float]]) -> list[int]:
    """
    Rank the 12 transformer encoder layers (indices 1-12 in layer_prefs correspond
    to encoder.layers[0-11]) by aggregated importance across tasks.

    Returns a list of transformer *encoder* indices (0-based, used in
    model.hubert.encoder.layers[i]), sorted best-first.
    """
    tasks = list(layer_prefs.keys())
    # Average importance across tasks for each encoder layer (skip index 0 = embed)
    avg = []
    for enc_idx in range(NUM_TRANSFORMER_LAYERS):
        hidden_state_idx = enc_idx + 1  # hidden_states[1] = encoder.layers[0]
        score = sum(layer_prefs[t][hidden_state_idx] for t in tasks) / len(tasks)
        avg.append((enc_idx, score))
    avg.sort(key=lambda x: -x[1])
    return [enc_idx for enc_idx, _ in avg]


def print_layer_analysis(layer_prefs: dict[str, list[float]], ranked: list[int]) -> None:
    """Pretty-print the layer importance analysis."""
    print("\n" + "=" * 60)
    print("  LAYER IMPORTANCE ANALYSIS")
    print("=" * 60)
    print("\nPer-task top-6 layers (layer 0 = feature projection embed,")
    print("layers 1-12 = transformer encoder layers 0-11):\n")
    for task, prefs in layer_prefs.items():
        ranked_task = sorted(enumerate(prefs), key=lambda x: -x[1])
        print(f"  {task.upper()}:")
        for rank, (idx, w) in enumerate(ranked_task[:6], 1):
            name = "feature_projection" if idx == 0 else f"encoder.layers[{idx - 1}]"
            print(f"    #{rank:2d}  hidden[{idx:2d}]  {name:<30s}  {w:.4f}")
    print()

    print("Aggregated ranking across all tasks (transformer layers only):\n")
    tasks = list(layer_prefs.keys())
    for rank, enc_idx in enumerate(ranked, 1):
        hidden_idx = enc_idx + 1
        scores = {t: layer_prefs[t][hidden_idx] for t in tasks}
        avg_score = sum(scores.values()) / len(scores)
        bar = "█" * int(avg_score * 200)
        print(f"  #{rank:2d}  encoder.layers[{enc_idx:2d}]  avg={avg_score:.4f}  "
              f"({', '.join(f'{t[0]}:{v:.3f}' for t, v in scores.items())})  {bar}")

    print("\nRecommended --unfreeze_top_n choices:")
    for n in [2, 4, 6, 8]:
        layers = ranked[:n]
        print(f"  top {n}: encoder.layers{sorted(layers)}")
    print("=" * 60 + "\n")


# ── Unfreezing helpers ────────────────────────────────────────────────────────

def unfreeze_layers(
    model: MultiTaskHubert,
    transformer_layer_indices: list[int],
    unfreeze_feature_proj: bool = False,
) -> list[nn.Parameter]:
    """
    Unfreeze specific transformer encoder layers (and optionally feature_projection).
    Returns the list of newly-unfrozen parameters.
    """
    unfrozen_params: list[nn.Parameter] = []

    if unfreeze_feature_proj:
        for p in model.hubert.feature_projection.parameters():
            p.requires_grad = True
            unfrozen_params.append(p)
        print("  Unfrozen: hubert.feature_projection")

    for idx in transformer_layer_indices:
        layer = model.hubert.encoder.layers[idx]
        for p in layer.parameters():
            p.requires_grad = True
            unfrozen_params.append(p)
        print(f"  Unfrozen: hubert.encoder.layers[{idx}]")

    return unfrozen_params


def describe_frozen_state(model: MultiTaskHubert) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,} / {total:,}  ({100 * trainable / total:.1f}%)")


# ── Main ─────────────────────────────────────────────────────────────────────

def build_optimizer(
    model: MultiTaskHubert,
    layer_indices: list[int],
    backbone_lr_top: float,
    layer_decay: float,
    head_lr: float,
) -> optim.Optimizer:
    """
    Build a discriminative-LR AdamW optimizer.

    Each unfrozen encoder layer n gets its own param group with a geometrically
    decayed learning rate:

        LR_n = backbone_lr_top * layer_decay^(NUM_TRANSFORMER_LAYERS - n)

    Layer 11 (closest to the output) is the least decayed; layer 0 the most.
    Heads and learnable layer-weights share a single group at head_lr.
    """
    param_groups: list[dict] = []

    m = _unwrap(model)

    # ── Per-layer backbone groups ──────────────────────────────────────────────
    for enc_idx in layer_indices:
        lr_n = backbone_lr_top * (layer_decay ** (NUM_TRANSFORMER_LAYERS - enc_idx))
        params = [p for p in m.hubert.encoder.layers[enc_idx].parameters()
                  if p.requires_grad]
        if params:
            param_groups.append({
                "params": params,
                "lr": lr_n,
                "name": f"encoder.layers[{enc_idx}]",
            })

    # Feature-projection (optional, unfrozen via --unfreeze_feature_proj)
    fp_params = [p for p in m.hubert.feature_projection.parameters() if p.requires_grad]
    if fp_params:
        # Use maximum decay: same LR as if it were encoder layer −1
        lr_fp = backbone_lr_top * (layer_decay ** (NUM_TRANSFORMER_LAYERS + 1))
        param_groups.append({"params": fp_params, "lr": lr_fp, "name": "feature_projection"})

    # ── Heads + learnable layer weights ───────────────────────────────────────
    head_params = (
        list(m.emotion_head.parameters())
        + list(m.gender_head.parameters())
        + list(m.age_head.parameters())
        + [m.emotion_weights, m.gender_weights, m.age_weights]
    )
    param_groups.append({"params": head_params, "lr": head_lr, "name": "heads"})

    _fused = torch.cuda.is_available()
    try:
        optimizer = optim.AdamW(param_groups, weight_decay=0.01, fused=_fused)
    except (TypeError, RuntimeError):
        optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    return optimizer


def _print_lr_schedule(layer_indices: list[int], backbone_lr_top: float,
                       layer_decay: float, head_lr: float) -> None:
    """Print a human-readable table of the per-layer LR assignments."""
    print("\n  Layer-wise LR schedule  (ξ = {:.2f}, LR_n = {:.2e} * ξ^(12-n))".format(
        layer_decay, backbone_lr_top))
    print(f"  {'Layer':<26}  {'Decay exp':>9}  {'LR':>10}")
    print("  " + "-" * 50)
    for enc_idx in sorted(layer_indices, reverse=True):
        exp = NUM_TRANSFORMER_LAYERS - enc_idx
        lr_n = backbone_lr_top * (layer_decay ** exp)
        print(f"  encoder.layers[{enc_idx:2d}]          {exp:9d}  {lr_n:10.2e}")
    print(f"  {'heads / layer-weights':<26}           -  {head_lr:10.2e}")
    print()


def main() -> None:
    global _stop_requested
    signal.signal(signal.SIGINT, _sigint_handler)
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Explicit CUDA throughput flags (also set at import time via train.py module,
    # but repeated here for clarity when running finetune.py standalone)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    # ── Layer analysis ────────────────────────────────────────────────────────
    # Use the last epoch of the most recent training run for ranking — this
    # reflects the final layer-weight preferences from stage-1 training.
    layer_prefs = latest_layer_prefs(MODEL_DIR)
    ranked_layers = rank_transformer_layers(layer_prefs)

    if ANALYZE_ONLY:
        print_layer_analysis(layer_prefs, ranked_layers)
        return

    print_layer_analysis(layer_prefs, ranked_layers)
    layers_to_unfreeze = ranked_layers[:UNFREEZE_TOP_N]
    print(f"Fine-tuning plan:")
    print(f"  Will unfreeze top-{UNFREEZE_TOP_N} transformer layers: "
          f"encoder.layers{sorted(layers_to_unfreeze)}")
    if UNFREEZE_FEATURE_PROJ:
        print("  Also unfreezing: feature_projection")

    # ── Load data ─────────────────────────────────────────────────────────────
    dataset, train_split, val_split, composition = build_mixed_train_val_splits()
    merged_for_encoders = {"train": train_split, "validation": val_split}
    print("Building label encoders...")
    emotion_encoder, gender_encoder, age_encoder = build_label_encoders(merged_for_encoders)
    num_emotions = len(emotion_encoder)
    num_genders = len(gender_encoder)
    num_ages = len(age_encoder)
    print(f"  {num_emotions} emotion | {num_genders} gender | {num_ages} age classes")

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    # Priority:
    #   1. models/finetune/latest_checkpoint_finetune.pt  (resume interrupted run)
    #   2. models/best_model.pt                           (fresh start from stage-1)
    metrics_path    = FINETUNE_DIR / "training_metrics_finetune.json"
    best_model_path = FINETUNE_DIR / "best_model_finetuned.pt"
    latest_ft_path  = FINETUNE_DIR / "latest_checkpoint_finetune.pt"
    step_ckpt_dir   = FINETUNE_DIR / "steps"

    _is_finetune_resume = False
    if latest_ft_path.exists():
        # Peek at the checkpoint to verify it is compatible and not finished
        _peek = torch.load(latest_ft_path, map_location="cpu", weights_only=False)
        if (_peek.get("num_emotions") == num_emotions
                and _peek.get("num_genders") == num_genders
                and _peek.get("num_ages") == num_ages
                and _peek["epoch"] + 1 < NUM_EPOCHS):
            ckpt_path = latest_ft_path
            _is_finetune_resume = True
        else:
            ckpt_path = MODEL_DIR / "best_model.pt"
    else:
        ckpt_path = MODEL_DIR / "best_model.pt"

    if not ckpt_path.exists():
        print("ERROR: No checkpoint found. Run train.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'Resuming finetune from' if _is_finetune_resume else 'Starting from'}: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Verify class counts match
    for key, expected in [("num_emotions", num_emotions), ("num_genders", num_genders),
                          ("num_ages", num_ages)]:
        if ckpt.get(key, expected) != expected:
            print(f"WARNING: checkpoint {key}={ckpt.get(key)} but dataset has {expected}. "
                  "Label encoders may have changed.", file=sys.stderr)

    # ── Build model ───────────────────────────────────────────────────────────
    model = MultiTaskHubert(
        num_emotions=num_emotions,
        num_genders=num_genders,
        num_ages=num_ages,
        freeze_backbone=True,  # Start fully frozen; we unfreeze selectively below
        use_spec_augment=True,  # Stage 2: online SpecAugment on hidden states
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    if hasattr(model.hubert.config, 'training_drop_path'):
        model.hubert.config.training_drop_path = 0.1

    # ── Unfreeze layers ───────────────────────────────────────────────────────
    unfreeze_layers(
        model,
        layers_to_unfreeze,
        unfreeze_feature_proj=UNFREEZE_FEATURE_PROJ,
    )

    describe_frozen_state(model)

    # ── Optimizer & metrics state ─────────────────────────────────────────────
    # Build optimizer before torch.compile: param tensors must be plain Python
    # objects so the optimizer holds direct references (not wrapped proxies).
    _print_lr_schedule(layers_to_unfreeze, BACKBONE_LR_TOP, LAYER_DECAY, HEAD_LR)
    optimizer = build_optimizer(model, layers_to_unfreeze,
                                BACKBONE_LR_TOP, LAYER_DECAY, HEAD_LR)
    # LR schedule: hold for 5 epochs, then cosine decay for 15 epochs to 1% of initial LR
    # Applies uniformly across all param groups, preserving their relative layer-decay ratios.
    # Use per-group decay shaping so high-LR groups can decay faster than low-LR groups
    # while still reaching the same eta_min_factor * base_lr endpoint.
    scheduler = _make_cosine_schedule(
        optimizer,
        hold_epochs=5,
        decay_epochs=15,
        scale_group_decay=True,
        group_power_range=(0.8, 1.3),
    )

    best_val_loss = float("inf")
    all_metrics: list[dict] = []
    last_epoch = -1
    start_epoch = 0
    epochs_without_improvement = 0
    step_state = {'global_step': 0, 'samples_seen': 0}
    wandb_run = None

    if _is_finetune_resume:
        # ckpt is already the finetune checkpoint — restore training state
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_epoch   = ckpt["epoch"] + 1
        step_state['global_step'] = int(ckpt.get('global_step', 0))
        step_state['samples_seen'] = int(ckpt.get('samples_seen', 0))
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            for _ in range(start_epoch):
                scheduler.step()
        if metrics_path.exists():
            with open(metrics_path) as f:
                all_metrics = json.load(f)
        print(f"Resuming fine-tuning from epoch {start_epoch + 1}/{NUM_EPOCHS}")

    if WANDB_ENABLED:
        if wandb is None:
            print("WARNING: WANDB_ENABLED=1 but wandb is not installed. Skipping W&B logging.")
        else:
            run_name = WANDB_RUN_NAME or f"stage2-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb_kwargs = {
                'project': WANDB_PROJECT,
                'name': run_name,
                'config': {
                    'batch_size': BATCH_SIZE,
                    'num_epochs': NUM_EPOCHS,
                    'backbone_lr_top': BACKBONE_LR_TOP,
                    'layer_decay': LAYER_DECAY,
                    'head_lr': HEAD_LR,
                    'emotion_weight': EMOTION_WEIGHT,
                    'gender_weight': GENDER_WEIGHT,
                    'age_weight': AGE_WEIGHT,
                    'unfreeze_top_n': UNFREEZE_TOP_N,
                    'unfreeze_feature_proj': UNFREEZE_FEATURE_PROJ,
                    'checkpoint_every_steps': CHECKPOINT_EVERY_STEPS,
                    'layers_to_unfreeze': sorted(layers_to_unfreeze),
                },
                'tags': WANDB_TAGS,
                'resume': 'allow',
            }
            if WANDB_ENTITY:
                wandb_kwargs['entity'] = WANDB_ENTITY
            if WANDB_RUN_GROUP:
                wandb_kwargs['group'] = WANDB_RUN_GROUP
            if WANDB_MODE:
                wandb_kwargs['mode'] = WANDB_MODE
            wandb_run = wandb.init(**wandb_kwargs)

    def _on_train_batch_end(payload):
        global_step = payload['global_step']
        if wandb_run is not None:
            data = {
                'train/loss_total': payload['train_total_loss'],
                'train/epoch': payload['epoch'] + 1,
            }
            if payload['train_emotion_loss'] is not None:
                data['train/loss_emotion'] = payload['train_emotion_loss']
            if payload['train_gender_loss'] is not None:
                data['train/loss_gender'] = payload['train_gender_loss']
            if payload['train_age_loss'] is not None:
                data['train/loss_age'] = payload['train_age_loss']
            wandb_run.log(data, step=global_step)

        if CHECKPOINT_EVERY_STEPS > 0 and global_step % CHECKPOINT_EVERY_STEPS == 0:
            step_path = _save_step_checkpoint(
                step_dir=step_ckpt_dir,
                global_step=global_step,
                epoch=payload['epoch'],
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                num_emotions=num_emotions,
                num_genders=num_genders,
                num_ages=num_ages,
                samples_seen=step_state['samples_seen'],
            )
            _rotate_step_checkpoints(step_ckpt_dir, CHECKPOINT_KEEP_LAST_N_STEP_FILES)
            print(f"Saved step checkpoint: {step_path.name}")

            if CHECKPOINT_SAVE_LATEST_EVERY_STEPS:
                torch.save({
                    'epoch': payload['epoch'],
                    'global_step': global_step,
                    'samples_seen': step_state['samples_seen'],
                    'model_state_dict': _unwrap(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'num_emotions': num_emotions,
                    'num_genders': num_genders,
                    'num_ages': num_ages,
                }, latest_ft_path)

            if wandb_run is not None and WANDB_UPLOAD_STEP_ARTIFACT:
                _save_wandb_file_artifact(
                    wandb_run,
                    file_path=step_path,
                    name=f"stage2-step-{global_step:08d}",
                    artifact_type='checkpoint',
                )

            if (wandb_run is not None and WANDB_UPLOAD_LATEST_ARTIFACT
                    and WANDB_LATEST_ARTIFACT_EVERY_STEPS > 0
                    and global_step % WANDB_LATEST_ARTIFACT_EVERY_STEPS == 0
                    and latest_ft_path.exists()):
                _save_wandb_file_artifact(
                    wandb_run,
                    file_path=latest_ft_path,
                    name=f"stage2-latest-{global_step:08d}",
                    artifact_type='checkpoint',
                )

    # torch.compile with mode='default' — see train.py for details.
    if device.type == 'cuda' and hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile(mode='default')...")
            model = torch.compile(model, mode='default', dynamic=False)
        except Exception as e:
            print(f"  torch.compile unavailable ({e}), running eager.")

    # ── Processor & datasets ───────────────────────────────────────────────────────
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

    print("Dataset composition:")
    print(f"  HF train/val: {composition['hf_train']} / {composition['hf_val']}")
    print(f"  Kazemo train/val: {composition['kazemo_train']} / {composition['kazemo_val']}")
    if composition.get("kazemo_emotion_counts"):
        emo_counts = ", ".join([f"{k}:{v}" for k, v in composition["kazemo_emotion_counts"].items()])
        print(f"  Kazemo selected emotion counts: {emo_counts}")
    print(f"  Mixed train/val total: {composition['train_total']} / {composition['val_total']}")
    print(f"  Train labels present: emotion={composition['train_label_counts']['emotion']}, "
          f"gender={composition['train_label_counts']['gender']}, "
          f"age={composition['train_label_counts']['age']}")
    print(f"  Val labels present: emotion={composition['val_label_counts']['emotion']}, "
          f"gender={composition['val_label_counts']['gender']}, "
          f"age={composition['val_label_counts']['age']}")

    train_dataset = AudioDataset(
        train_split,
        processor,
        emotion_encoder,
        gender_encoder,
        age_encoder,
        is_train=True,
        noise_dir=os.environ.get("NOISE_DIR"),
    )
    val_dataset = AudioDataset(
        val_split,
        processor,
        emotion_encoder,
        gender_encoder,
        age_encoder,
        is_train=False,
        noise_dir=None,
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=pin,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=pin,
                              persistent_workers=True, prefetch_factor=2)

    criterion_emotion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_gender  = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_age     = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\nStarting fine-tuning for {NUM_EPOCHS} epochs. Press Ctrl+C to stop and save.")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS):
        if _stop_requested:
            break

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)

        train_metrics, stopped = train_epoch(
            model, train_loader, criterion_emotion, criterion_gender,
            criterion_age, optimizer, device,
            step_state=step_state,
            on_batch_end=_on_train_batch_end,
            epoch_index=epoch,
        )
        last_epoch = epoch
        print(f"Train Loss - Total: {train_metrics['total']:.4f},  "
              f"Emotion: {train_metrics['emotion']:.4f},  "
              f"Gender: {train_metrics['gender']:.4f},  "
              f"Age: {train_metrics['age']:.4f}")
        if stopped:
            break

        val_metrics = validate(model, val_loader, criterion_emotion, criterion_gender,
                               criterion_age, device)
        if wandb_run is not None:
            wandb_run.log({
                'val/loss_total': float(val_metrics['total']),
                'val/loss_emotion': float(val_metrics['emotion']),
                'val/loss_gender': float(val_metrics['gender']),
                'val/loss_age': float(val_metrics['age']),
                'val/acc_emotion': float(val_metrics['emotion_acc']),
                'val/acc_gender': float(val_metrics['gender_acc']),
                'val/acc_age': float(val_metrics['age_acc']),
                'val/epoch': epoch + 1,
            }, step=step_state['global_step'])
        if _stop_requested:
            break

        print(f"Val   Loss - Total: {val_metrics['total']:.4f},  "
              f"Emotion: {val_metrics['emotion']:.4f},  "
              f"Gender: {val_metrics['gender']:.4f},  "
              f"Age: {val_metrics['age']:.4f}")
        print(f"Val   Acc  - Emotion: {val_metrics['emotion_acc']:.4f},  "
              f"Gender: {val_metrics['gender_acc']:.4f},  "
              f"Age: {val_metrics['age_acc']:.4f}")

        def lp(w):  # layer prefs helper
            return [round(x, 6) for x in torch.softmax(w, dim=0).detach().cpu().tolist()]
        _m = _unwrap(model)
        epoch_record = {
            "epoch": epoch + 1,
            "unfrozen_layers": sorted(layers_to_unfreeze),
            "train": {k: round(float(v), 6) for k, v in train_metrics.items()},
            "val":   {k: round(float(v), 6) for k, v in val_metrics.items()},
            "layer_prefs": {
                "emotion": lp(_m.emotion_weights),
                "gender":  lp(_m.gender_weights),
                "age":     lp(_m.age_weights),
            },
        }
        all_metrics.append(epoch_record)
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                'global_step': step_state['global_step'],
                'samples_seen': step_state['samples_seen'],
                "model_state_dict": _unwrap(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "num_emotions": num_emotions,
                "num_genders": num_genders,
                "num_ages": num_ages,
                "unfrozen_layers": sorted(layers_to_unfreeze),
            }, best_model_path)
            print(f"  ✓ New best val loss: {best_val_loss:.4f} → saved to {best_model_path.name}")
            if wandb_run is not None and WANDB_UPLOAD_BEST_ARTIFACT:
                _save_wandb_file_artifact(
                    wandb_run,
                    file_path=best_model_path,
                    name=f"stage2-best-epoch-{epoch + 1}",
                    artifact_type='model',
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping: no val loss improvement for {EARLY_STOPPING_PATIENCE} epochs.")
                break

        # Advance LR schedule and log current LRs (top backbone layer / head)
        scheduler.step()
        print(f"  LR backbone/head: {optimizer.param_groups[0]['lr']:.2e} / {optimizer.param_groups[-1]['lr']:.2e}")
        if wandb_run is not None:
            lr_values = [float(g['lr']) for g in optimizer.param_groups]
            wandb_run.log({
                'train/lr_backbone_top': float(optimizer.param_groups[0]['lr']),
                'train/lr_head': float(optimizer.param_groups[-1]['lr']),
                'train/lr_mean': float(sum(lr_values) / max(len(lr_values), 1)),
                'train/lr_min': float(min(lr_values)),
                'train/lr_max': float(max(lr_values)),
            }, step=step_state['global_step'])

        torch.save({
            "epoch": epoch,
            'global_step': step_state['global_step'],
            'samples_seen': step_state['samples_seen'],
            "model_state_dict": _unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "num_emotions": num_emotions,
            "num_genders": num_genders,
            "num_ages": num_ages,
        }, latest_ft_path)
        if wandb_run is not None and WANDB_UPLOAD_LATEST_ARTIFACT and WANDB_LATEST_ARTIFACT_EVERY_STEPS <= 0:
            _save_wandb_file_artifact(
                wandb_run,
                file_path=latest_ft_path,
                name=f"stage2-latest-epoch-{epoch + 1}",
                artifact_type='checkpoint',
            )

    # ── Save interrupted checkpoint ───────────────────────────────────────────
    if _stop_requested:
        int_path = FINETUNE_DIR / "checkpoint_finetune_interrupted.pt"
        torch.save({
            "epoch": last_epoch,
            'global_step': step_state['global_step'],
            'samples_seen': step_state['samples_seen'],
            "model_state_dict": _unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_emotions": num_emotions,
            "num_genders": num_genders,
            "num_ages": num_ages,
        }, int_path)
        print(f"\nStopped by user. Checkpoint saved to {int_path}")
    else:
        print("\nFine-tuning complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
