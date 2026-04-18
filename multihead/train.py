#!/usr/bin/env python3
"""Fine-tune HuBERT with multiple heads (emotion, gender, age) on audio data."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
import signal
import warnings
import logging

# Disable torchcodec and use soundfile for audio decoding
os.environ['DATASETS_AUDIO_BACKEND'] = 'soundfile'
os.environ['TORCHCODEC_QUIET'] = '1'
os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '0'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='torch._dynamo')
warnings.filterwarnings('ignore', module='torch._inductor.utils')
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor

try:
    import wandb
except Exception:
    wandb = None

import utils.misc as _utils_misc
from multihead.model import MultiTaskHubert
from utils.checkpointing import (
    MODEL_DIR,
    make_cosine_schedule,
    rotate_step_checkpoints,
    save_step_checkpoint,
    save_wandb_file_artifact,
)
from utils.data import AudioDataset, VAL_FRACTION, build_label_encoders, fallback_split_train_val
from utils.data_loading import (
    HF_BATCH01_CACHE,
    HF_BATCH01_ID,
    HF_BATCH01_SPLIT,
    HF_BATCH02_CACHE,
    HF_BATCH02_ID,
    KAZEMO_MAX_SAMPLES,
    TRAIN_VAL_MANIFEST,
    USE_KAZEMO,
    build_holdout_mixed_train_val_splits,
    build_mixed_train_val_splits,
)
from utils.config import load_stage_config
from utils.misc import (
    BATCH_SIZE_ENV_VAR,
    RANDOM_SEED,
    REPO_ROOT,
    SAMPLE_RATE,
    _ALL_TASKS,
    apply_cuda_perf_flags,
    set_seed,
    sigint_handler,
    unwrap,
)
from utils.training import (
    _wandb_val_keys,
    filter_val_metrics,
    make_batch_end_handler,
    train_epoch,
    validate,
)

CFG = load_stage_config(1)
_tcfg = CFG["training"]
BATCH_SIZE = CFG["runtime"]["batch_size"]
HEAD_LEARNING_RATE = float(_tcfg["head_learning_rate"])
NUM_EPOCHS = int(_tcfg["num_epochs"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
apply_cuda_perf_flags(DEVICE)

EMOTION_WEIGHT = float(_tcfg["emotion_weight"])
GENDER_WEIGHT = float(_tcfg["gender_weight"])
AGE_WEIGHT = float(_tcfg["age_weight"])
GRAD_CLIP_NORM = float(_tcfg["grad_clip_norm"])
EARLY_STOPPING_PATIENCE = int(_tcfg["early_stopping_patience"])
LABEL_SMOOTHING = float(_tcfg.get("label_smoothing", 0.1))
WEIGHT_DECAY = float(_tcfg.get("weight_decay", 0.01))
_SCHED = _tcfg.get("scheduler", {})

_RT = CFG["runtime"]
NUM_CHECKPOINTS = _RT["num_checkpoints"]
CHECKPOINT_EVERY_STEPS = _RT["checkpoint_every_steps"]
CHECKPOINT_KEEP_LAST_N_STEP_FILES = _RT["keep_last_n"]
CHECKPOINT_SAVE_LATEST_EVERY_STEPS = _RT["save_latest_every_steps"]
NUM_VALIDATIONS = _RT["num_validations"]
VAL_EVERY_STEPS = _RT["val_every_steps"]

_WB = _RT["wandb"]
WANDB_ENABLED = _WB["enabled"]
WANDB_ENTITY = _WB["entity"]
WANDB_PROJECT = _WB["project"]
WANDB_RUN_NAME = _WB["run_name"]
WANDB_RUN_GROUP = _WB["run_group"]
WANDB_MODE = _WB["mode"]
WANDB_TAGS = _WB["tags"]
WANDB_UPLOAD_BEST_ARTIFACT = _WB["upload_best"]
WANDB_UPLOAD_LATEST_ARTIFACT = _WB["upload_latest"]
WANDB_UPLOAD_STEP_ARTIFACT = _WB["upload_step"]
WANDB_LATEST_ARTIFACT_EVERY_STEPS = _WB["latest_every_steps"]


def main():
    set_seed(RANDOM_SEED)
    signal.signal(signal.SIGINT, sigint_handler)
    print(f"Using device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE} ({BATCH_SIZE_ENV_VAR} env override)")
    print("Press Ctrl+C to stop training and save a checkpoint.")

    _dataset, train_split, val_split, named_val_splits, composition = build_mixed_train_val_splits()
    named_val_tasks: dict = composition.get("named_val_tasks", {})
    merged_for_encoders = DatasetDict({"train": train_split, "validation": val_split})

    print("Building label encoders...")
    emotion_encoder, gender_encoder, age_encoder = build_label_encoders(merged_for_encoders)
    num_emotions = len(emotion_encoder)
    num_genders = len(gender_encoder)
    num_ages = len(age_encoder)

    print(f"Found {num_emotions} emotion classes: {list(emotion_encoder.keys())}")
    print(f"Found {num_genders} gender classes: {list(gender_encoder.keys())}")
    print(f"Found {num_ages} age classes: {list(age_encoder.keys())}")

    encoder_path = MODEL_DIR / "label_encoders.json"
    with open(encoder_path, 'w') as f:
        json.dump({
            'emotion': emotion_encoder,
            'gender': gender_encoder,
            'age': age_encoder,
        }, f, indent=2)
    print(f"Saved label encoders to {encoder_path}")

    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

    print("Dataset composition:")
    _mode = composition.get("mode")
    if _mode == "split_manifest":
        print(f"  Mode: split_manifest ({composition['manifest_dir']})")
        print(f"  Test total: {composition.get('test_total', 0)}")
    elif _mode == "holdout_manifest":
        print(f"  Mode: holdout_manifest (val = validate.py subset only, no leakage into train)")
        print(f"  Manifest: {composition['manifest']}")
        print(
            f"  batch01 train-only / val holdout: {composition['batch01_train_only']} / {composition['hf_val']}"
        )
        print(f"  batch02 train-only (full split): {composition['batch02_train_only']}")
    if "hf_train" in composition:
        print(f"  HF train/val: {composition['hf_train']} / {composition['hf_val']}")
    if "kazemo_train" in composition:
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
        is_train=True, noise_dir=CFG["runtime"]["noise_dir"],
    )
    _pin = DEVICE.type == 'cuda'
    _dl = CFG.get("dataloader", {})
    _nw = int(_dl.get("num_workers", 4))
    _pf = int(_dl.get("prefetch_factor", 2))
    _pw = bool(_dl.get("persistent_workers", True))
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=_nw, pin_memory=_pin, persistent_workers=_pw, prefetch_factor=_pf,
    )
    val_loaders = {
        name: DataLoader(
            AudioDataset(split, processor, emotion_encoder, gender_encoder, age_encoder,
                         is_train=False, noise_dir=None),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=_nw, pin_memory=_pin, persistent_workers=_pw, prefetch_factor=_pf,
        )
        for name, split in named_val_splits.items()
    }
    _primary_val = next(iter(val_loaders))

    print("Initializing model...")
    _mcfg = CFG.get("model", {})
    model = MultiTaskHubert(
        num_emotions=num_emotions,
        num_genders=num_genders,
        num_ages=num_ages,
        freeze_backbone=bool(_mcfg.get("freeze_backbone", True)),
        use_spec_augment=bool(_mcfg.get("use_spec_augment", False)),
    ).to(DEVICE)

    head_params = (
        list(model.emotion_head.parameters()) +
        list(model.gender_head.parameters()) +
        list(model.age_head.parameters()) +
        [model.emotion_weights, model.gender_weights, model.age_weights]
    )

    _fused_opt = DEVICE.type == 'cuda'
    try:
        optimizer = optim.AdamW([{'params': head_params, 'lr': HEAD_LEARNING_RATE}], weight_decay=WEIGHT_DECAY, fused=_fused_opt)
        if _fused_opt:
            print("  Optimizer: fused AdamW")
    except (TypeError, RuntimeError):
        optimizer = optim.AdamW([{'params': head_params, 'lr': HEAD_LEARNING_RATE}], weight_decay=WEIGHT_DECAY)
        print("  Optimizer: standard AdamW (fused not available)")

    scheduler = make_cosine_schedule(
        optimizer,
        hold_epochs=int(_SCHED.get("hold_epochs", 3)),
        decay_epochs=int(_SCHED.get("decay_epochs", 7)),
    )

    criterion_emotion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    criterion_gender = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    criterion_age = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    metrics_path = MODEL_DIR / "training_metrics.json"
    latest_path = MODEL_DIR / "latest_checkpoint.pt"
    step_ckpt_dir = MODEL_DIR / "steps"
    train_state = {'best_val_loss': float('inf')}
    all_metrics = []
    all_step_val_metrics: list[dict] = []
    step_val_metrics_path = MODEL_DIR / "step_val_metrics.json"
    if step_val_metrics_path.exists():
        with open(step_val_metrics_path) as f:
            all_step_val_metrics = json.load(f)
    start_epoch = 0
    epochs_without_improvement = 0
    step_state = {'global_step': 0, 'samples_seen': 0}
    wandb_run = None

    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=DEVICE, weights_only=False)
        if (ckpt.get("num_emotions") == num_emotions and ckpt.get("num_genders") == num_genders
                and ckpt.get("num_ages") == num_ages
                and ckpt["epoch"] + 1 < NUM_EPOCHS):
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            train_state['best_val_loss'] = ckpt.get("best_val_loss", float("inf"))
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
            print(f"Resuming from epoch {start_epoch + 1}/{NUM_EPOCHS} (latest checkpoint)")

    if WANDB_ENABLED:
        if wandb is None:
            print("WARNING: WANDB_ENABLED=1 but wandb is not installed. Skipping W&B logging.")
        else:
            run_name = WANDB_RUN_NAME or f"stage1-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb_kwargs = {
                'project': WANDB_PROJECT,
                'name': run_name,
                'config': {
                    'batch_size': BATCH_SIZE,
                    'num_epochs': NUM_EPOCHS,
                    'head_learning_rate': HEAD_LEARNING_RATE,
                    'emotion_weight': EMOTION_WEIGHT,
                    'gender_weight': GENDER_WEIGHT,
                    'age_weight': AGE_WEIGHT,
                    'grad_clip_norm': GRAD_CLIP_NORM,
                    'early_stopping_patience': EARLY_STOPPING_PATIENCE,
                    'use_kazemo': USE_KAZEMO,
                    'checkpoint_every_steps': CHECKPOINT_EVERY_STEPS,
                    'val_every_steps': VAL_EVERY_STEPS,
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

    _total_steps = max(1, (NUM_EPOCHS - start_epoch) * len(train_loader))
    _ckpt_every = CHECKPOINT_EVERY_STEPS
    if _ckpt_every == 0 and NUM_CHECKPOINTS > 0:
        _ckpt_every = max(1, _total_steps // NUM_CHECKPOINTS)
        print(f"  Checkpointing: {NUM_CHECKPOINTS} evenly-spaced (every {_ckpt_every} steps of {_total_steps})")
    _val_every = VAL_EVERY_STEPS
    if _val_every == 0 and NUM_VALIDATIONS > 0:
        _val_every = max(1, _total_steps // NUM_VALIDATIONS)
        print(f"  Step validations: {NUM_VALIDATIONS} evenly-spaced (every {_val_every} steps)")

    _on_train_batch_end = make_batch_end_handler(
        step_state=step_state,
        train_state=train_state,
        all_step_val_metrics=all_step_val_metrics,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        num_emotions=num_emotions,
        num_genders=num_genders,
        num_ages=num_ages,
        checkpoint_every_steps=_ckpt_every,
        checkpoint_keep_last_n=CHECKPOINT_KEEP_LAST_N_STEP_FILES,
        checkpoint_save_latest_every_steps=CHECKPOINT_SAVE_LATEST_EVERY_STEPS,
        step_ckpt_dir=step_ckpt_dir,
        latest_path=latest_path,
        step_val_metrics_path=step_val_metrics_path,
        val_every_steps=_val_every,
        val_loaders=val_loaders,
        val_tasks=named_val_tasks,
        criterion_emotion=criterion_emotion,
        criterion_gender=criterion_gender,
        criterion_age=criterion_age,
        device=DEVICE,
        emotion_weight=EMOTION_WEIGHT,
        gender_weight=GENDER_WEIGHT,
        age_weight=AGE_WEIGHT,
        wandb_run=wandb_run,
        wandb_upload_step_artifact=WANDB_UPLOAD_STEP_ARTIFACT,
        wandb_upload_latest_artifact=WANDB_UPLOAD_LATEST_ARTIFACT,
        wandb_latest_artifact_every_steps=WANDB_LATEST_ARTIFACT_EVERY_STEPS,
        step_artifact_name="stage1-step",
        latest_artifact_name="stage1-latest",
    )

    _ccfg = CFG.get("compile", {})
    if bool(_ccfg.get("enabled", True)) and DEVICE.type == 'cuda' and hasattr(torch, 'compile'):
        _cmode = str(_ccfg.get("mode", "default"))
        try:
            print(f"Compiling model with torch.compile(mode='{_cmode}')...")
            model = torch.compile(model, mode=_cmode, dynamic=None)
        except Exception as e:
            print(f"  torch.compile unavailable ({e}), running eager.")

    for epoch in range(start_epoch, NUM_EPOCHS):
        if _utils_misc.stop_requested:
            break
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)

        train_metrics, stopped = train_epoch(
            model, train_loader, criterion_emotion, criterion_gender,
            criterion_age, optimizer, DEVICE,
            step_state=step_state,
            on_batch_end=_on_train_batch_end,
            epoch_index=epoch,
            emotion_weight=EMOTION_WEIGHT,
            gender_weight=GENDER_WEIGHT,
            age_weight=AGE_WEIGHT,
            grad_clip_norm=GRAD_CLIP_NORM,
        )

        print(f"Train Loss - Total: {train_metrics['total']:.4f}, "
              f"Emotion: {train_metrics['emotion']:.4f}, "
              f"Gender: {train_metrics['gender']:.4f}, "
              f"Age: {train_metrics['age']:.4f}")
        print(f"Train Acc  - Emotion: {train_metrics['emotion_acc']:.4f}, "
              f"Gender: {train_metrics['gender_acc']:.4f}, "
              f"Age: {train_metrics['age_acc']:.4f}")
        if stopped:
            break

        all_val_metrics: dict = {}
        wandb_epoch_data: dict = {
            'train/acc_emotion': float(train_metrics['emotion_acc']),
            'train/acc_gender': float(train_metrics['gender_acc']),
            'train/acc_age': float(train_metrics['age_acc']),
        }
        for _name, _vloader in val_loaders.items():
            _prefix = "val" if _name == "val" else f"val_{_name}"
            _tasks = named_val_tasks.get(_name, _ALL_TASKS)
            _vm = filter_val_metrics(validate(
                model, _vloader, criterion_emotion, criterion_gender,
                criterion_age, DEVICE,
                emotion_weight=EMOTION_WEIGHT,
                gender_weight=GENDER_WEIGHT,
                age_weight=AGE_WEIGHT,
            ), _tasks)
            all_val_metrics[_name] = _vm
            wandb_epoch_data.update(_wandb_val_keys(_prefix, _vm))
            wandb_epoch_data[f'{_prefix}/epoch'] = epoch + 1
            _loss_line = (f"Val [{_name}] Loss: {_vm['total']:.4f}  "
                          f"Emotion: {_vm['emotion']:.4f}")
            if "gender" in _tasks:
                _loss_line += f"  Gender: {_vm['gender']:.4f}"
            if "age" in _tasks:
                _loss_line += f"  Age: {_vm['age']:.4f}"
            print(_loss_line)
            _acc_line = f"Val [{_name}] Acc:  Emotion: {_vm['emotion_acc']:.4f}"
            if "gender" in _tasks:
                _acc_line += f"  Gender: {_vm['gender_acc']:.4f}"
            if "age" in _tasks:
                _acc_line += f"  Age: {_vm['age_acc']:.4f}"
            print(_acc_line)
        if wandb_run is not None:
            wandb_run.log(wandb_epoch_data, step=step_state['global_step'])
        if _utils_misc.stop_requested:
            break

        val_metrics = all_val_metrics[_primary_val]  # primary split drives early stopping / best model

        def layer_prefs_tolist(weights):
            return torch.softmax(weights, dim=0).detach().cpu().tolist()

        epoch_record: dict = {
            "epoch": epoch + 1,
            "train": {k: round(float(v), 6) for k, v in train_metrics.items()},
            "layer_prefs": {
                "emotion": [round(x, 6) for x in layer_prefs_tolist(unwrap(model).emotion_weights)],
                "gender": [round(x, 6) for x in layer_prefs_tolist(unwrap(model).gender_weights)],
                "age": [round(x, 6) for x in layer_prefs_tolist(unwrap(model).age_weights)],
            },
        }
        for _name, _vm in all_val_metrics.items():
            _key = "val" if _name == "val" else f"val_{_name}"
            epoch_record[_key] = {k: round(float(v), 6) for k, v in _vm.items()}
        all_metrics.append(epoch_record)
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")

        if val_metrics['total'] < train_state['best_val_loss']:
            train_state['best_val_loss'] = val_metrics['total']
            epochs_without_improvement = 0
            model_path = MODEL_DIR / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'global_step': step_state['global_step'],
                'samples_seen': step_state['samples_seen'],
                'model_state_dict': unwrap(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': train_state['best_val_loss'],
                'num_emotions': num_emotions,
                'num_genders': num_genders,
                'num_ages': num_ages,
            }, model_path)
            print(f"Saved best model to {model_path}")
            if wandb_run is not None and WANDB_UPLOAD_BEST_ARTIFACT:
                save_wandb_file_artifact(
                    wandb_run,
                    file_path=model_path,
                    name="stage1-best",
                    artifact_type='model',
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping: no val loss improvement for {EARLY_STOPPING_PATIENCE} epochs.")
                break

        scheduler.step()
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        torch.save({
            'epoch': epoch,
            'global_step': step_state['global_step'],
            'samples_seen': step_state['samples_seen'],
            'model_state_dict': unwrap(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': train_state['best_val_loss'],
            'num_emotions': num_emotions,
            'num_genders': num_genders,
            'num_ages': num_ages,
        }, latest_path)
        if wandb_run is not None and WANDB_UPLOAD_LATEST_ARTIFACT and WANDB_LATEST_ARTIFACT_EVERY_STEPS <= 0:
            save_wandb_file_artifact(
                wandb_run,
                file_path=latest_path,
                name="stage1-latest",
                artifact_type='checkpoint',
            )

    if _utils_misc.stop_requested:
        print("\nStopped by user.")
    else:
        print("\nTraining completed!")
    print(f"Best validation loss: {train_state['best_val_loss']:.4f}")
    if wandb_run is not None:
        wandb_run.finish()


# ── Backward-compatible aliases ─────────────────────────────────────────────
_make_cosine_schedule = make_cosine_schedule
_unwrap = unwrap
_sigint_handler = sigint_handler
_save_wandb_file_artifact = save_wandb_file_artifact
_save_step_checkpoint = save_step_checkpoint
_rotate_step_checkpoints = rotate_step_checkpoints

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
