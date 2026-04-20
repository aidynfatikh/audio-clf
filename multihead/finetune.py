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
from torch.utils.data import DataLoader

try:
    import wandb
except Exception:
    wandb = None

import utils.misc as _utils_misc
from multihead.model import MultiTaskBackbone
from utils.checkpointing import (
    MODEL_DIR,
    make_cosine_schedule,
    save_wandb_file_artifact,
)
from utils.data import AudioDataset, build_label_encoders, compute_class_weights
from utils.data_loading import (
    KAZEMO_MAX_SAMPLES,
    USE_KAZEMO,
    build_mixed_train_val_splits,
)
from utils.finetune_utils import (
    build_optimizer,
    describe_frozen_state,
    latest_layer_prefs,
    print_layer_analysis,
    rank_transformer_layers,
    unfreeze_layers,
)
from utils.finetune_utils import print_lr_schedule as _print_lr_schedule
from utils.config import build_feature_extractor, load_stage_config
from utils.misc import (
    BATCH_SIZE_ENV_VAR,
    RANDOM_SEED,
    _ALL_TASKS,
    apply_cuda_perf_flags,
    set_seed,
    sigint_handler,
    unwrap,
)
from utils.training import (
    _wandb_val_keys,
    evaluate_and_save_test_results,
    filter_val_metrics,
    make_batch_end_handler,
    train_epoch,
    validate,
)

CFG = load_stage_config(2)
_tcfg = CFG["training"]
_ftcfg = CFG.get("finetune", {})
_mcfg = CFG.get("model", {})
_SCHED = _tcfg.get("scheduler", {})

BATCH_SIZE = CFG["runtime"]["batch_size"]
NUM_EPOCHS = int(_tcfg["num_epochs"])

BACKBONE_LR_TOP = float(_ftcfg.get("backbone_lr_top", 5e-5))
LAYER_DECAY = float(_ftcfg.get("layer_decay", 0.85))
HEAD_LR = float(_ftcfg.get("head_lr", _tcfg.get("head_learning_rate", 2e-4)))

EMOTION_WEIGHT = float(_tcfg["emotion_weight"])
GENDER_WEIGHT = float(_tcfg["gender_weight"])
AGE_WEIGHT = float(_tcfg["age_weight"])
GRAD_CLIP_NORM = float(_tcfg["grad_clip_norm"])
LABEL_SMOOTHING = float(_tcfg.get("label_smoothing", 0.1))
CLASS_WEIGHTING = str(_tcfg.get("class_weighting", "none")).strip().lower()

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FINETUNE_DIR = MODEL_DIR / "finetune"
FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

UNFREEZE_TOP_N = int(_ftcfg.get("unfreeze_top_n", 4))
UNFREEZE_FEATURE_PROJ = bool(_ftcfg.get("unfreeze_feature_proj", False))
TRAINING_DROP_PATH = float(_ftcfg.get("training_drop_path", 0.1))
INIT_FROM = str(_ftcfg.get("init_from", "") or "")
ANALYZE_ONLY = "--analyze" in sys.argv
EARLY_STOPPING_ENABLED = bool(_tcfg.get("early_stopping", False))
EARLY_STOPPING_PATIENCE = int(_tcfg["early_stopping_patience"])

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


def main() -> None:
    signal.signal(signal.SIGINT, sigint_handler)
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Batch size: {BATCH_SIZE} ({BATCH_SIZE_ENV_VAR} env override)")
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

    _, train_split, val_split, named_val_splits, composition = build_mixed_train_val_splits()
    named_val_tasks: dict = composition.get("named_val_tasks", {})
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

    _default_init = Path(INIT_FROM) if INIT_FROM else (MODEL_DIR / "best_model.pt")
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
            ckpt_path = _default_init
    else:
        ckpt_path = _default_init

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

    _backbone_name = str(_mcfg.get("name", "hubert_base"))
    _pretrained = str(_mcfg.get("pretrained", "facebook/hubert-base-ls960"))
    print(f"Backbone: {_backbone_name} ({_pretrained})")
    model = MultiTaskBackbone(
        num_emotions=num_emotions,
        num_genders=num_genders,
        num_ages=num_ages,
        freeze_backbone=bool(_mcfg.get("freeze_backbone", True)),
        use_spec_augment=bool(_mcfg.get("use_spec_augment", True)),
        backbone_name=_backbone_name,
        pretrained=_pretrained,
    ).to(device)
    _ckpt_extra = {"backbone_name": _backbone_name, "pretrained": _pretrained}
    model.load_state_dict(ckpt["model_state_dict"])

    if hasattr(model.hubert.config, 'training_drop_path'):
        model.hubert.config.training_drop_path = TRAINING_DROP_PATH

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
        hold_epochs=int(_SCHED.get("hold_epochs", 5)),
        decay_epochs=int(_SCHED.get("decay_epochs", 15)),
        scale_group_decay=bool(_SCHED.get("scale_group_decay", True)),
        group_power_range=(
            float(_SCHED.get("group_power_lo", 0.8)),
            float(_SCHED.get("group_power_hi", 1.3)),
        ),
    )

    train_state = {'best_val_loss': float("inf")}
    all_metrics: list[dict] = []
    all_step_val_metrics: list[dict] = []
    step_val_metrics_path = FINETUNE_DIR / "step_val_metrics_finetune.json"
    step_train_metrics_path = FINETUNE_DIR / "step_train_metrics_finetune.jsonl"
    if step_val_metrics_path.exists():
        with open(step_val_metrics_path) as f:
            all_step_val_metrics = json.load(f)
    start_epoch = 0
    epochs_without_improvement = 0
    step_state = {'global_step': 0, 'samples_seen': 0}
    wandb_run = None

    if _is_finetune_resume:
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
                    'val_every_steps': VAL_EVERY_STEPS,
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

    _ccfg = CFG.get("compile", {})
    if bool(_ccfg.get("enabled", True)) and device.type == 'cuda' and hasattr(torch, 'compile'):
        _cmode = str(_ccfg.get("mode", "default"))
        try:
            print(f"Compiling model with torch.compile(mode='{_cmode}')...")
            model = torch.compile(model, mode=_cmode, dynamic=None)
        except Exception as e:
            print(f"  torch.compile unavailable ({e}), running eager.")

    processor = build_feature_extractor(_pretrained)

    print("Dataset composition:")
    _mode = composition.get("mode")
    if _mode == "split_manifest":
        print(f"  Mode: split_manifest ({composition['manifest_dir']})")
        print(f"  Test total: {composition.get('test_total', 0)}")
    elif _mode == "holdout_manifest":
        print("  Mode: holdout_manifest (val = validate.py subset only, no leakage into train)")
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

    pin = device.type == "cuda"
    _dl = CFG.get("dataloader", {})
    _nw = int(_dl.get("num_workers", 4))
    _pf = int(_dl.get("prefetch_factor", 2))
    _pw = bool(_dl.get("persistent_workers", True))
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=_nw, pin_memory=pin, persistent_workers=_pw, prefetch_factor=_pf,
    )
    val_loaders = {
        name: DataLoader(
            AudioDataset(split, processor, emotion_encoder, gender_encoder, age_encoder,
                         is_train=False, noise_dir=None),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=_nw, pin_memory=pin, persistent_workers=_pw, prefetch_factor=_pf,
        )
        for name, split in named_val_splits.items()
    }
    # Per-corpus val (batch01/batch02/kazemo); best-model tracks the macro-average.

    _cls_w: dict[str, torch.Tensor] | None = None
    if CLASS_WEIGHTING == "balanced":
        _raw_w = compute_class_weights(
            train_split,
            {"emotion": emotion_encoder, "gender": gender_encoder, "age": age_encoder},
        )
        _cls_w = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in _raw_w.items()}
        for task, enc in (("emotion", emotion_encoder), ("gender", gender_encoder), ("age", age_encoder)):
            inv_enc = {i: lbl for lbl, i in enc.items()}
            pairs = ", ".join(f"{inv_enc[i]}={w:.3f}" for i, w in enumerate(_raw_w[task]))
            print(f"  Class weights [{task}]: {pairs}")
    elif CLASS_WEIGHTING not in ("none", ""):
        raise ValueError(f"training.class_weighting must be 'balanced' or 'none', got {CLASS_WEIGHTING!r}")

    criterion_emotion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING,
                                            weight=_cls_w["emotion"] if _cls_w else None)
    criterion_gender = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING,
                                           weight=_cls_w["gender"] if _cls_w else None)
    criterion_age = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING,
                                        weight=_cls_w["age"] if _cls_w else None)

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
        latest_path=latest_ft_path,
        step_val_metrics_path=step_val_metrics_path,
        step_train_metrics_path=step_train_metrics_path,
        val_every_steps=_val_every,
        val_loaders=val_loaders,
        val_tasks=named_val_tasks,
        criterion_emotion=criterion_emotion,
        criterion_gender=criterion_gender,
        criterion_age=criterion_age,
        device=device,
        emotion_weight=EMOTION_WEIGHT,
        gender_weight=GENDER_WEIGHT,
        age_weight=AGE_WEIGHT,
        wandb_run=wandb_run,
        wandb_upload_step_artifact=WANDB_UPLOAD_STEP_ARTIFACT,
        wandb_upload_latest_artifact=WANDB_UPLOAD_LATEST_ARTIFACT,
        wandb_latest_artifact_every_steps=WANDB_LATEST_ARTIFACT_EVERY_STEPS,
        step_artifact_name="stage2-step",
        latest_artifact_name="stage2-latest",
        ckpt_extra=_ckpt_extra,
    )

    print(f"\nStarting fine-tuning for {NUM_EPOCHS} epochs. Press Ctrl+C to stop and save.")

    for epoch in range(start_epoch, NUM_EPOCHS):
        if _utils_misc.stop_requested:
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
        print(f"Train Loss - Total: {train_metrics['total']:.4f},  "
              f"Emotion: {train_metrics['emotion']:.4f},  "
              f"Gender: {train_metrics['gender']:.4f},  "
              f"Age: {train_metrics['age']:.4f}")
        print(f"Train Acc  - Emotion: {train_metrics['emotion_acc']:.4f},  "
              f"Gender: {train_metrics['gender_acc']:.4f},  "
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
            _prefix = f"val_{_name}"
            _tasks = named_val_tasks.get(_name, _ALL_TASKS)
            _vm = filter_val_metrics(validate(
                model, _vloader, criterion_emotion, criterion_gender,
                criterion_age, device,
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
        _vals = list(all_val_metrics.values())
        _macro_total = sum(m["total"] for m in _vals) / max(len(_vals), 1)
        val_metrics = {"total": _macro_total}
        wandb_epoch_data['val_macro/loss_total'] = float(_macro_total)
        print(f"Val [macro] Loss: {_macro_total:.4f}  (average of {len(_vals)} corpus splits)")
        if wandb_run is not None:
            wandb_run.log(wandb_epoch_data, step=step_state['global_step'])
        if _utils_misc.stop_requested:
            break

        def lp(w):
            return [round(x, 6) for x in torch.softmax(w, dim=0).detach().cpu().tolist()]
        _m = unwrap(model)
        epoch_record: dict = {
            "epoch": epoch + 1,
            "unfrozen_layers": sorted(layers_to_unfreeze),
            "train": {k: round(float(v), 6) for k, v in train_metrics.items()},
            "layer_prefs": {
                "emotion": lp(_m.emotion_weights),
                "gender": lp(_m.gender_weights),
                "age": lp(_m.age_weights),
            },
        }
        for _name, _vm in all_val_metrics.items():
            epoch_record[f"val_{_name}"] = {k: round(float(v), 6) for k, v in _vm.items()}
        epoch_record["val_macro"] = {"total": round(float(_macro_total), 6)}
        all_metrics.append(epoch_record)
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        if val_metrics["total"] < train_state['best_val_loss']:
            train_state['best_val_loss'] = val_metrics["total"]
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                'global_step': step_state['global_step'],
                'samples_seen': step_state['samples_seen'],
                "model_state_dict": unwrap(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": train_state['best_val_loss'],
                "num_emotions": num_emotions,
                "num_genders": num_genders,
                "num_ages": num_ages,
                "unfrozen_layers": sorted(layers_to_unfreeze),
                **_ckpt_extra,
            }, best_model_path)
            print(f"  ✓ New best val loss: {train_state['best_val_loss']:.4f} → saved to {best_model_path.name}")
        else:
            epochs_without_improvement += 1
            if EARLY_STOPPING_ENABLED and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
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
            "best_val_loss": train_state['best_val_loss'],
            "num_emotions": num_emotions,
            "num_genders": num_genders,
            "num_ages": num_ages,
            **_ckpt_extra,
        }, latest_ft_path)

    if _utils_misc.stop_requested:
        print("\nStopped by user.")
    else:
        print("\nFine-tuning complete!")
    print(f"Best validation loss: {train_state['best_val_loss']:.4f}")

    if not _utils_misc.stop_requested and composition.get("mode") == "split_manifest":
        try:
            from splits.materialize import materialize_named_test
            _test_per_corpus = materialize_named_test(Path(composition["manifest_dir"]))
            _test_tasks = {n: set(_ALL_TASKS) for n in _test_per_corpus}
            if "kazemo" in _test_tasks:
                from utils.training import _KAZEMO_TASKS
                _test_tasks["kazemo"] = set(_KAZEMO_TASKS)
            evaluate_and_save_test_results(
                model=model,
                test_splits=_test_per_corpus,
                test_tasks=_test_tasks,
                processor=processor,
                emotion_encoder=emotion_encoder,
                gender_encoder=gender_encoder,
                age_encoder=age_encoder,
                criterion_emotion=criterion_emotion,
                criterion_gender=criterion_gender,
                criterion_age=criterion_age,
                device=device,
                batch_size=BATCH_SIZE,
                num_workers=_nw,
                prefetch_factor=_pf,
                persistent_workers=_pw,
                pin_memory=pin,
                emotion_weight=EMOTION_WEIGHT,
                gender_weight=GENDER_WEIGHT,
                age_weight=AGE_WEIGHT,
                out_path=FINETUNE_DIR / "test_results_finetune.json",
                best_ckpt_path=best_model_path,
                label="stage2",
                wandb_run=wandb_run,
            )
        except Exception as e:
            print(f"[stage2] Test eval failed: {e}")

    if wandb_run is not None:
        if WANDB_UPLOAD_BEST_ARTIFACT and best_model_path.exists():
            save_wandb_file_artifact(wandb_run, file_path=best_model_path, name="stage2-best", artifact_type='model')
        if WANDB_UPLOAD_LATEST_ARTIFACT and latest_ft_path.exists():
            save_wandb_file_artifact(wandb_run, file_path=latest_ft_path, name="stage2-latest", artifact_type='checkpoint')
        wandb_run.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
