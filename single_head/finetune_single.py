#!/usr/bin/env python3
"""
Fine-tune SingleHeadHubert by partially unfreezing backbone layers.

Loads a checkpoint from train_single.py for the given feature, unfreezes
top-N transformer layers (ranked by that feature's layer weights), and continues training.

Usage:
  python finetune_single.py                    # Fine-tune emotion (default)
  python finetune_single.py --feature age      # Fine-tune age model
  python finetune_single.py --feature gender
  python finetune_single.py --analyze          # Print layer importance and exit
"""

import argparse
import os
import signal
import sys
import warnings
from pathlib import Path

# Ensure repo root is on path when running from single_head/
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["TORCHCODEC_QUIET"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor
from datasets import Audio
import json

from load_data import load
from train_single import (
    SingleHeadHubert,
    AudioDataset,
    build_label_encoders,
    train_epoch,
    validate,
    _sigint_handler,
    _unwrap,
    _make_cosine_schedule,
    FEATURES,
)

BATCH_SIZE = 8
NUM_EPOCHS = 20
SAMPLE_RATE = 16000
BACKBONE_LR_TOP = 5e-5
LAYER_DECAY = 0.85
HEAD_LR = 2e-4

NUM_HUBERT_LAYERS = 13
NUM_TRANSFORMER_LAYERS = 12

_stop_requested = False


def load_layer_prefs(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        data = json.load(f)
    if not data:
        return None
    return data[-1].get("layer_prefs")


def latest_layer_prefs(metrics_dir: Path, feature: str) -> list:
    """Return layer preferences for the given feature from the last epoch of training_metrics.json."""
    path = metrics_dir / "training_metrics.json"
    prefs = load_layer_prefs(path)
    if prefs is None or feature not in prefs:
        return [1.0 / NUM_HUBERT_LAYERS] * NUM_HUBERT_LAYERS
    raw = prefs[feature]
    total = sum(raw) or 1.0
    return [v / total for v in raw]


def rank_transformer_layers(layer_prefs: list) -> list[int]:
    """Rank transformer layers (indices 1–12) by importance; return encoder indices best-first."""
    avg = []
    for enc_idx in range(NUM_TRANSFORMER_LAYERS):
        hidden_state_idx = enc_idx + 1
        score = layer_prefs[hidden_state_idx] if hidden_state_idx < len(layer_prefs) else 0.0
        avg.append((enc_idx, score))
    avg.sort(key=lambda x: -x[1])
    return [enc_idx for enc_idx, _ in avg]


def print_layer_analysis(layer_prefs: list, ranked: list[int], feature: str) -> None:
    print("\n" + "=" * 60)
    print(f"  LAYER IMPORTANCE ({feature})")
    print("=" * 60)
    ranked_task = sorted(enumerate(layer_prefs), key=lambda x: -x[1])
    print("\nTop-6 layers:")
    for rank, (idx, w) in enumerate(ranked_task[:6], 1):
        name = "feature_projection" if idx == 0 else f"encoder.layers[{idx - 1}]"
        print(f"  #{rank:2d}  hidden[{idx:2d}]  {name:<30s}  {w:.4f}")
    print("\nRanking (transformer layers):")
    for rank, enc_idx in enumerate(ranked, 1):
        hi = enc_idx + 1
        sc = layer_prefs[hi] if hi < len(layer_prefs) else 0.0
        print(f"  #{rank:2d}  encoder.layers[{enc_idx:2d}]  {sc:.4f}")
    print("=" * 60 + "\n")


def unfreeze_layers(
    model: SingleHeadHubert,
    transformer_layer_indices: list[int],
    unfreeze_feature_proj: bool = False,
) -> list:
    unfrozen = []
    if unfreeze_feature_proj:
        for p in model.hubert.feature_projection.parameters():
            p.requires_grad = True
            unfrozen.append(p)
        print("  Unfrozen: hubert.feature_projection")
    for idx in transformer_layer_indices:
        for p in model.hubert.encoder.layers[idx].parameters():
            p.requires_grad = True
            unfrozen.append(p)
        print(f"  Unfrozen: hubert.encoder.layers[{idx}]")
    return unfrozen


def describe_frozen_state(model: SingleHeadHubert) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,} / {total:,}  ({100 * trainable / total:.1f}%)")


def build_optimizer(
    model: SingleHeadHubert,
    layer_indices: list[int],
    backbone_lr_top: float,
    layer_decay: float,
    head_lr: float,
) -> optim.Optimizer:
    param_groups = []
    m = _unwrap(model)
    for enc_idx in layer_indices:
        lr_n = backbone_lr_top * (layer_decay ** (NUM_TRANSFORMER_LAYERS - enc_idx))
        params = [p for p in m.hubert.encoder.layers[enc_idx].parameters() if p.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": lr_n, "name": f"encoder.layers[{enc_idx}]"})
    fp_params = [p for p in m.hubert.feature_projection.parameters() if p.requires_grad]
    if fp_params:
        lr_fp = backbone_lr_top * (layer_decay ** (NUM_TRANSFORMER_LAYERS + 1))
        param_groups.append({"params": fp_params, "lr": lr_fp, "name": "feature_projection"})
    head_params = list(m.head.parameters()) + [m.layer_weights]
    param_groups.append({"params": head_params, "lr": head_lr, "name": "head"})
    try:
        return optim.AdamW(param_groups, weight_decay=0.01, fused=torch.cuda.is_available())
    except (TypeError, RuntimeError):
        return optim.AdamW(param_groups, weight_decay=0.01)


def main() -> None:
    global _stop_requested
    parser = argparse.ArgumentParser(description="Fine-tune single-head HuBERT.")
    parser.add_argument("--feature", default="emotion", choices=list(FEATURES), help="Feature to finetune.")
    parser.add_argument("--analyze", action="store_true", help="Print layer importance and exit.")
    parser.add_argument("--unfreeze_top_n", type=int, default=4, help="Number of top layers to unfreeze.")
    args = parser.parse_args()
    feature = args.feature
    ANALYZE_ONLY = args.analyze
    UNFREEZE_TOP_N = args.unfreeze_top_n

    signal.signal(signal.SIGINT, _sigint_handler)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    MODEL_BASE = Path(__file__).resolve().parent / "models"
    MODEL_DIR = MODEL_BASE / feature
    FINETUNE_DIR = MODEL_DIR / "finetune"
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

    layer_prefs = latest_layer_prefs(MODEL_DIR, feature)
    ranked_layers = rank_transformer_layers(layer_prefs)
    if ANALYZE_ONLY:
        print_layer_analysis(layer_prefs, ranked_layers, feature)
        return

    print_layer_analysis(layer_prefs, ranked_layers, feature)
    layers_to_unfreeze = ranked_layers[:UNFREEZE_TOP_N]
    print(f"Fine-tuning plan: unfreeze top-{UNFREEZE_TOP_N} layers: encoder.layers{sorted(layers_to_unfreeze)}")

    dataset = load()
    print("Building label encoders...")
    emotion_encoder, gender_encoder, age_encoder = build_label_encoders(dataset)
    encoders = {"emotion": emotion_encoder, "gender": gender_encoder, "age": age_encoder}
    num_classes = len(encoders[feature])
    print(f"  {feature}: {num_classes} classes")

    metrics_path = FINETUNE_DIR / "training_metrics_finetune.json"
    best_model_path = FINETUNE_DIR / "best_model_finetuned.pt"
    latest_ft_path = FINETUNE_DIR / "latest_checkpoint_finetune.pt"
    stage1_path = MODEL_DIR / "best_model.pt"

    _is_finetune_resume = False
    if latest_ft_path.exists():
        _peek = torch.load(latest_ft_path, map_location="cpu", weights_only=False)
        if (_peek.get("feature") == feature and _peek.get("num_classes") == num_classes
                and _peek["epoch"] + 1 < NUM_EPOCHS):
            _is_finetune_resume = True
            ckpt_path = latest_ft_path
        else:
            ckpt_path = stage1_path
    else:
        ckpt_path = stage1_path

    if not ckpt_path.exists():
        print("ERROR: No checkpoint found. Run train_single.py --feature", feature, "first.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'Resuming finetune from' if _is_finetune_resume else 'Starting from'}: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = SingleHeadHubert(feature=feature, num_classes=num_classes, freeze_backbone=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    unfreeze_layers(model, layers_to_unfreeze, unfreeze_feature_proj=False)
    describe_frozen_state(model)

    optimizer = build_optimizer(model, layers_to_unfreeze, BACKBONE_LR_TOP, LAYER_DECAY, HEAD_LR)
    # LR schedule: hold for 5 epochs, then cosine decay for 15 epochs to 1% of initial LR
    # Applies uniformly across all param groups, preserving their relative layer-decay ratios.
    scheduler = _make_cosine_schedule(optimizer, hold_epochs=5, decay_epochs=15)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    all_metrics = []
    last_epoch = -1
    start_epoch = 0
    if _is_finetune_resume:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_epoch = ckpt["epoch"] + 1
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            for _ in range(start_epoch):
                scheduler.step()
        if metrics_path.exists():
            with open(metrics_path) as f:
                all_metrics = json.load(f)
        print(f"Resuming from epoch {start_epoch + 1}/{NUM_EPOCHS}")

    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            print("Compiling model with torch.compile(mode='default')...")
            model = torch.compile(model, mode="default", dynamic=False)
        except Exception as e:
            print(f"  torch.compile unavailable ({e}), running eager.")

    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    train_split = dataset.get("train") or dataset[list(dataset.keys())[0]]
    val_split = dataset.get("validation") or dataset.get("val") or dataset.get("test")
    if "audio" in train_split.column_names:
        train_split = train_split.cast_column("audio", Audio(decode=False))
    if val_split is not None and "audio" in val_split.column_names:
        val_split = val_split.cast_column("audio", Audio(decode=False))

    train_dataset = AudioDataset(train_split, processor, emotion_encoder, gender_encoder, age_encoder)
    val_dataset = AudioDataset(
        val_split or train_split, processor,
        emotion_encoder, gender_encoder, age_encoder,
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=pin, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=pin, persistent_workers=True, prefetch_factor=2)

    print(f"\nStarting fine-tuning for {NUM_EPOCHS} epochs [{feature}]. Press Ctrl+C to stop.")

    for epoch in range(start_epoch, NUM_EPOCHS):
        if _stop_requested:
            break
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} [{feature}]")
        print("-" * 50)
        train_metrics, stopped = train_epoch(
            model, train_loader, criterion, optimizer, device, feature
        )
        last_epoch = epoch
        print(f"Train Loss: {train_metrics['total']:.4f}")
        if stopped:
            break
        val_metrics = validate(model, val_loader, criterion, device, feature)
        if _stop_requested:
            break
        print(f"Val Loss: {val_metrics['total']:.4f}  Val Acc: {val_metrics['acc']:.4f}")

        def lp(w):
            return [round(x, 6) for x in torch.softmax(w, dim=0).detach().cpu().tolist()]
        _m = _unwrap(model)
        epoch_record = {
            "epoch": epoch + 1,
            "unfrozen_layers": sorted(layers_to_unfreeze),
            "train": {k: round(float(v), 6) for k, v in train_metrics.items()},
            "val": {k: round(float(v), 6) for k, v in val_metrics.items()},
            "layer_prefs": {feature: lp(_m.layer_weights)},
        }
        all_metrics.append(epoch_record)
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": _unwrap(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "feature": feature,
                "num_classes": num_classes,
                "unfrozen_layers": sorted(layers_to_unfreeze),
            }, best_model_path)
            print(f"  ✓ New best val loss: {best_val_loss:.4f} → {best_model_path.name}")

        # Advance LR schedule and log current LRs (top backbone layer / head)
        scheduler.step()
        print(f"  LR backbone/head: {optimizer.param_groups[0]['lr']:.2e} / {optimizer.param_groups[-1]['lr']:.2e}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": _unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "feature": feature,
            "num_classes": num_classes,
        }, latest_ft_path)

    if _stop_requested:
        int_path = FINETUNE_DIR / "checkpoint_finetune_interrupted.pt"
        torch.save({
            "epoch": last_epoch,
            "model_state_dict": _unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "feature": feature,
            "num_classes": num_classes,
        }, int_path)
        print(f"\nStopped by user. Checkpoint saved to {int_path}")
    else:
        print("\nFine-tuning complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
