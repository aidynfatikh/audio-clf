#!/usr/bin/env python3
"""
Fine-tune MultiTaskHubert by partially unfreezing backbone layers.

Stage 2 training: loads a checkpoint from frozen-backbone training (``multihead/train.py``),
then unfreezes the top-N HuBERT transformer layers (ranked by learned per-task
layer weights), and continues training with a low backbone LR.

Data loading uses utils.build_mixed_train_val_splits(): set TRAIN_VAL_MANIFEST and
HF_BATCH01_*/HF_BATCH02_* the same as for stage-1 training (val = validate.py holdout only).

Usage (from repo root or from this directory):
  python multihead/finetune.py
  python finetune.py              # if cwd is multihead/
  python finetune.py --analyze    # layer importance only
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
import signal
import warnings
from datetime import datetime

os.environ['DATASETS_AUDIO_BACKEND'] = 'soundfile'
os.environ['TORCHCODEC_QUIET'] = '1'
warnings.filterwarnings('ignore', category=UserWarning)

import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor

try:
    import wandb
except Exception:
    wandb = None

import multihead.utils as utils
from utils.finetune_utils import (
    build_optimizer,
    describe_frozen_state,
    latest_layer_prefs,
    print_layer_analysis,
    rank_transformer_layers,
    unfreeze_layers,
)
from utils.finetune_utils import print_lr_schedule as _print_lr_schedule
from multihead.model import MultiTaskHubert
from multihead.utils import (
    AudioDataset,
    KAZEMO_MAX_SAMPLES,
    MODEL_DIR,
    RANDOM_SEED,
    USE_KAZEMO,
    apply_cuda_perf_flags,
    build_label_encoders,
    build_mixed_train_val_splits,
    make_cosine_schedule,
    rotate_step_checkpoints,
    save_step_checkpoint,
    save_wandb_file_artifact,
    set_seed,
    sigint_handler,
    train_epoch,
    unwrap,
    validate,
)

BATCH_SIZE = 8
NUM_EPOCHS = 20

BACKBONE_LR_TOP = 5e-5
LAYER_DECAY = 0.85
HEAD_LR = 2e-4

EMOTION_WEIGHT = 1.2
GENDER_WEIGHT = 0.5
AGE_WEIGHT = 1.0
GRAD_CLIP_NORM = 1.0

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FINETUNE_DIR = MODEL_DIR / "finetune"
FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

UNFREEZE_TOP_N = 4
UNFREEZE_FEATURE_PROJ = False
ANALYZE_ONLY = "--analyze" in sys.argv
EARLY_STOPPING_PATIENCE = 5

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


def main() -> None:
    signal.signal(signal.SIGINT, sigint_handler)
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    apply_cuda_perf_flags(device)

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

    _dataset, train_split, val_split, composition = build_mixed_train_val_splits()
    merged_for_encoders = {"train": train_split, "validation": val_split}
    print("Building label encoders...")
    emotion_encoder, gender_encoder, age_encoder = build_label_encoders(merged_for_encoders)
    num_emotions = len(emotion_encoder)
    num_genders = len(gender_encoder)
    num_ages = len(age_encoder)
    print(f"  {num_emotions} emotion | {num_genders} gender | {num_ages} age classes")

    metrics_path = FINETUNE_DIR / "training_metrics_finetune.json"
    best_model_path = FINETUNE_DIR / "best_model_finetuned.pt"
    latest_ft_path = FINETUNE_DIR / "latest_checkpoint_finetune.pt"
    step_ckpt_dir = FINETUNE_DIR / "steps"

    _is_finetune_resume = False
    if latest_ft_path.exists():
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
        print("ERROR: No checkpoint found. Run multihead/train.py (stage 1) first.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'Resuming finetune from' if _is_finetune_resume else 'Starting from'}: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    for key, expected in [("num_emotions", num_emotions), ("num_genders", num_genders),
                          ("num_ages", num_ages)]:
        if ckpt.get(key, expected) != expected:
            print(f"WARNING: checkpoint {key}={ckpt.get(key)} but dataset has {expected}. "
                  "Label encoders may have changed.", file=sys.stderr)

    model = MultiTaskHubert(
        num_emotions=num_emotions,
        num_genders=num_genders,
        num_ages=num_ages,
        freeze_backbone=True,
        use_spec_augment=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    if hasattr(model.hubert.config, 'training_drop_path'):
        model.hubert.config.training_drop_path = 0.1

    unfreeze_layers(
        model,
        layers_to_unfreeze,
        unfreeze_feature_proj=UNFREEZE_FEATURE_PROJ,
    )

    describe_frozen_state(model)

    _print_lr_schedule(layers_to_unfreeze, BACKBONE_LR_TOP, LAYER_DECAY, HEAD_LR)
    optimizer = build_optimizer(model, layers_to_unfreeze,
                                BACKBONE_LR_TOP, LAYER_DECAY, HEAD_LR)
    scheduler = make_cosine_schedule(
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
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_epoch = ckpt["epoch"] + 1
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
            step_path = save_step_checkpoint(
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
            rotate_step_checkpoints(step_ckpt_dir, CHECKPOINT_KEEP_LAST_N_STEP_FILES)
            print(f"Saved step checkpoint: {step_path.name}")

            if CHECKPOINT_SAVE_LATEST_EVERY_STEPS:
                torch.save({
                    'epoch': payload['epoch'],
                    'global_step': global_step,
                    'samples_seen': step_state['samples_seen'],
                    'model_state_dict': unwrap(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'num_emotions': num_emotions,
                    'num_genders': num_genders,
                    'num_ages': num_ages,
                }, latest_ft_path)

            if wandb_run is not None and WANDB_UPLOAD_STEP_ARTIFACT:
                save_wandb_file_artifact(
                    wandb_run,
                    file_path=step_path,
                    name="stage2-step",
                    artifact_type='checkpoint',
                )

            if (wandb_run is not None and WANDB_UPLOAD_LATEST_ARTIFACT
                    and WANDB_LATEST_ARTIFACT_EVERY_STEPS > 0
                    and global_step % WANDB_LATEST_ARTIFACT_EVERY_STEPS == 0
                    and latest_ft_path.exists()):
                save_wandb_file_artifact(
                    wandb_run,
                    file_path=latest_ft_path,
                    name="stage2-latest",
                    artifact_type='checkpoint',
                )

    if device.type == 'cuda' and hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile(mode='default')...")
            model = torch.compile(model, mode='default', dynamic=None)
        except Exception as e:
            print(f"  torch.compile unavailable ({e}), running eager.")

    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

    print("Dataset composition:")
    if composition.get("mode") == "holdout_manifest":
        print("  Mode: holdout_manifest (val = validate.py subset only, no leakage into train)")
        print(f"  Manifest: {composition['manifest']}")
        print(
            f"  batch01 train-only / val holdout: {composition['batch01_train_only']} / {composition['hf_val']}"
        )
        print(f"  batch02 train-only (full split): {composition['batch02_train_only']}")
    print(f"  HF train/val: {composition['hf_train']} / {composition['hf_val']}")
    print(
        f"  Kazemo train/val (cap={KAZEMO_MAX_SAMPLES}, enabled={USE_KAZEMO}): "
        f"{composition['kazemo_train']} / {composition['kazemo_val']}"
    )
    if composition.get("kazemo_emotion_counts"):
        emo_counts = ", ".join([f"{k}:{v}" for k, v in composition["kazemo_emotion_counts"].items()])
        print(f"  Kazemo selected emotion counts: {emo_counts}")
    print(f"  Mixed train/val total: {composition['train_total']} / {composition['val_total']}")
    print(
        f"  Train labels present: emotion={composition['train_label_counts']['emotion']}, "
        f"gender={composition['train_label_counts']['gender']}, "
        f"age={composition['train_label_counts']['age']}"
    )
    print(
        f"  Val labels present: emotion={composition['val_label_counts']['emotion']}, "
        f"gender={composition['val_label_counts']['gender']}, "
        f"age={composition['val_label_counts']['age']}"
    )

    train_dataset = AudioDataset(
        train_split, processor, emotion_encoder, gender_encoder, age_encoder,
        is_train=True, noise_dir=os.environ.get("NOISE_DIR"),
    )
    val_dataset = AudioDataset(
        val_split, processor, emotion_encoder, gender_encoder, age_encoder,
        is_train=False, noise_dir=None,
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=pin, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=pin, persistent_workers=True, prefetch_factor=2,
    )

    criterion_emotion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_gender = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_age = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\nStarting fine-tuning for {NUM_EPOCHS} epochs. Press Ctrl+C to stop and save.")

    for epoch in range(start_epoch, NUM_EPOCHS):
        if utils.stop_requested:
            break

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)

        train_metrics, stopped = train_epoch(
            model, train_loader, criterion_emotion, criterion_gender,
            criterion_age, optimizer, device,
            step_state=step_state,
            on_batch_end=_on_train_batch_end,
            epoch_index=epoch,
            emotion_weight=EMOTION_WEIGHT,
            gender_weight=GENDER_WEIGHT,
            age_weight=AGE_WEIGHT,
            grad_clip_norm=GRAD_CLIP_NORM,
        )
        last_epoch = epoch
        print(f"Train Loss - Total: {train_metrics['total']:.4f},  "
              f"Emotion: {train_metrics['emotion']:.4f},  "
              f"Gender: {train_metrics['gender']:.4f},  "
              f"Age: {train_metrics['age']:.4f}")
        print(f"Train Acc  - Emotion: {train_metrics['emotion_acc']:.4f},  "
              f"Gender: {train_metrics['gender_acc']:.4f},  "
              f"Age: {train_metrics['age_acc']:.4f}")
        if stopped:
            break

        val_metrics = validate(
            model, val_loader, criterion_emotion, criterion_gender,
            criterion_age, device,
            emotion_weight=EMOTION_WEIGHT,
            gender_weight=GENDER_WEIGHT,
            age_weight=AGE_WEIGHT,
        )
        if wandb_run is not None:
            wandb_run.log({
                'train/acc_emotion': float(train_metrics['emotion_acc']),
                'train/acc_gender': float(train_metrics['gender_acc']),
                'train/acc_age': float(train_metrics['age_acc']),
                'val/loss_total': float(val_metrics['total']),
                'val/loss_emotion': float(val_metrics['emotion']),
                'val/loss_gender': float(val_metrics['gender']),
                'val/loss_age': float(val_metrics['age']),
                'val/acc_emotion': float(val_metrics['emotion_acc']),
                'val/acc_gender': float(val_metrics['gender_acc']),
                'val/acc_age': float(val_metrics['age_acc']),
                'val/epoch': epoch + 1,
            }, step=step_state['global_step'])
        if utils.stop_requested:
            break

        print(f"Val   Loss - Total: {val_metrics['total']:.4f},  "
              f"Emotion: {val_metrics['emotion']:.4f},  "
              f"Gender: {val_metrics['gender']:.4f},  "
              f"Age: {val_metrics['age']:.4f}")
        print(f"Val   Acc  - Emotion: {val_metrics['emotion_acc']:.4f},  "
              f"Gender: {val_metrics['gender_acc']:.4f},  "
              f"Age: {val_metrics['age_acc']:.4f}")

        def lp(w):
            return [round(x, 6) for x in torch.softmax(w, dim=0).detach().cpu().tolist()]
        _m = unwrap(model)
        epoch_record = {
            "epoch": epoch + 1,
            "unfrozen_layers": sorted(layers_to_unfreeze),
            "train": {k: round(float(v), 6) for k, v in train_metrics.items()},
            "val": {k: round(float(v), 6) for k, v in val_metrics.items()},
            "layer_prefs": {
                "emotion": lp(_m.emotion_weights),
                "gender": lp(_m.gender_weights),
                "age": lp(_m.age_weights),
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
                "model_state_dict": unwrap(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "num_emotions": num_emotions,
                "num_genders": num_genders,
                "num_ages": num_ages,
                "unfrozen_layers": sorted(layers_to_unfreeze),
            }, best_model_path)
            print(f"  ✓ New best val loss: {best_val_loss:.4f} → saved to {best_model_path.name}")
            if wandb_run is not None and WANDB_UPLOAD_BEST_ARTIFACT:
                save_wandb_file_artifact(
                    wandb_run,
                    file_path=best_model_path,
                    name="stage2-best",
                    artifact_type='model',
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping: no val loss improvement for {EARLY_STOPPING_PATIENCE} epochs.")
                break

        scheduler.step()
        print(f"  LR backbone/head: {optimizer.param_groups[0]['lr']:.2e} / {optimizer.param_groups[-1]['lr']:.2e}")

        torch.save({
            "epoch": epoch,
            'global_step': step_state['global_step'],
            'samples_seen': step_state['samples_seen'],
            "model_state_dict": unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "num_emotions": num_emotions,
            "num_genders": num_genders,
            "num_ages": num_ages,
        }, latest_ft_path)
        if wandb_run is not None and WANDB_UPLOAD_LATEST_ARTIFACT and WANDB_LATEST_ARTIFACT_EVERY_STEPS <= 0:
            save_wandb_file_artifact(
                wandb_run,
                file_path=latest_ft_path,
                name="stage2-latest",
                artifact_type='checkpoint',
            )

    if utils.stop_requested:
        int_path = FINETUNE_DIR / "checkpoint_finetune_interrupted.pt"
        torch.save({
            "epoch": last_epoch,
            'global_step': step_state['global_step'],
            'samples_seen': step_state['samples_seen'],
            "model_state_dict": unwrap(model).state_dict(),
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
