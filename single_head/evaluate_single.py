#!/usr/bin/env python3
"""Evaluate single-head HuBERT checkpoints and plot training history.

Outputs written to single_head/eval/ (or single_head/eval/{feature}/ if you want per-feature):
  training_curves.png   — loss + accuracy (stage-1 → finetune)
  layer_weights.png     — learned layer attention for the feature
  eval_results.json     — per-checkpoint accuracy + per-class breakdown
  accuracy_bars.png     — per-class accuracy

Usage:
  python single_head/evaluate_single.py --feature emotion   # default
  python single_head/evaluate_single.py --feature age --split test
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import Audio
from tqdm import tqdm

from load_data import load
from train_single import AudioDataset, build_label_encoders, FEATURES
from inference_single import load_model

MODEL_BASE = Path(__file__).resolve().parent / "models"
EVAL_DIR = Path(__file__).resolve().parent / "eval"


def _load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _series(epochs: list, key_path: list):
    out = []
    for e in epochs:
        node = e
        for k in key_path:
            node = node.get(k, {})
        out.append(float(node) if isinstance(node, (int, float)) else float("nan"))
    return out


def plot_training_curves(s1: list, ft: list, feature: str, out_path: Path) -> None:
    if not s1 and not ft:
        print("  No training metrics — skipping training curves.")
        return
    n_s1 = len(s1)
    x_s1 = list(range(1, n_s1 + 1))
    x_ft = list(range(n_s1 + 1, n_s1 + len(ft) + 1))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    fig.suptitle(f"Training History: {feature} (single-head)", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(x_s1, _series(s1, ["train", "total"]), "-", color="#4C72B0", lw=2, label="Train (stage-1)")
    ax.plot(x_s1, _series(s1, ["val", "total"]), "--", color="#4C72B0", lw=2, label="Val (stage-1)")
    ax.plot(x_ft, _series(ft, ["train", "total"]), "-", color="#DD8452", lw=2, label="Train (finetune)")
    ax.plot(x_ft, _series(ft, ["val", "total"]), "--", color="#DD8452", lw=2, label="Val (finetune)")
    if n_s1 and s1:
        best_idx = int(np.nanargmin(_series(s1, ["val", "total"])))
        ax.axvline(s1[best_idx]["epoch"], color="grey", ls=":", alpha=0.7)
    if n_s1 and ft:
        ax.axvline(n_s1 + 0.5, color="black", ls="-.", lw=1, alpha=0.5)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(x_s1, _series(s1, ["val", "acc"]), "-", color="#4C72B0", lw=2, label="Val acc (stage-1)")
    ax.plot(x_ft, _series(ft, ["val", "acc"]), "-", color="#DD8452", lw=2, label="Val acc (finetune)")
    if n_s1 and s1:
        best_idx = int(np.nanargmin(_series(s1, ["val", "total"])))
        ax.axvline(s1[best_idx]["epoch"], color="grey", ls=":", alpha=0.7)
    if n_s1 and ft:
        ax.axvline(n_s1 + 0.5, color="black", ls="-.", lw=1, alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("Val Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_layer_weights(s1: list, ft: list, feature: str, out_path: Path) -> None:
    if not s1 and not ft:
        return
    n_layers = 13
    n_s1 = len(s1)
    x_s1 = list(range(1, n_s1 + 1))
    x_ft = list(range(n_s1 + 1, n_s1 + len(ft) + 1))
    cmap = plt.cm.tab20
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
    ax.set_title(f"Layer attention: {feature}", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Softmax weight")
    ax.grid(True, alpha=0.2)
    for li in range(n_layers):
        colour = cmap(li / n_layers)
        lbl = "CNN" if li == 0 else f"L{li}"
        def _layer_series(epochs):
            return [
                e["layer_prefs"][feature][li]
                if ("layer_prefs" in e and feature in e["layer_prefs"] and li < len(e["layer_prefs"][feature]))
                else float("nan")
                for e in epochs
            ]
        ws_s1 = _layer_series(s1)
        ws_ft = _layer_series(ft)
        if any(not np.isnan(w) for w in ws_s1):
            ax.plot(x_s1, ws_s1, color=colour, lw=1.5, alpha=0.9, label=lbl)
        if any(not np.isnan(w) for w in ws_ft):
            ax.plot(x_ft, ws_ft, color=colour, lw=1.5, ls="--", alpha=0.9)
    if n_s1 and ft:
        ax.axvline(n_s1 + 0.5, color="black", ls="-.", lw=1, alpha=0.5)
    ax.legend(fontsize=6, ncol=4)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_accuracy_bars(all_results: dict, feature: str, out_path: Path) -> None:
    """all_results: {model_label: {split_name: res}}. res has accuracy[feature], per_class[feature], support[feature]."""
    model_labels = sorted(all_results.keys())
    n_models = len(model_labels)
    split_colours = ["#4C72B0", "#DD8452", "#55A868"]
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), constrained_layout=True)
    if n_models == 1:
        axes = [axes]
    fig.suptitle(f"Per-class accuracy: {feature} (single-head)", fontsize=14, fontweight="bold")
    for col, mlabel in enumerate(model_labels):
        ax = axes[col]
        ax.set_title(mlabel, fontsize=10)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.15)
        split_names = sorted(all_results[mlabel].keys())
        n_splits = len(split_names)
        group_w = 0.75
        bar_w = group_w / max(n_splits, 1)
        all_classes = sorted({
            cls
            for sname in split_names
            for cls in all_results[mlabel][sname]["per_class"][feature]
        })
        x_base = np.arange(len(all_classes))
        for si, sname in enumerate(split_names):
            res = all_results[mlabel][sname]
            offset = (si - (n_splits - 1) / 2) * bar_w
            vals = [res["per_class"][feature].get(c, 0.0) for c in all_classes]
            ax.bar(x_base + offset, vals, bar_w * 0.92, label=sname,
                   color=split_colours[si % len(split_colours)], alpha=0.85, edgecolor="white")
            ov = res["accuracy"][feature]
            ax.axhline(ov, color=split_colours[si % len(split_colours)], ls="--", lw=1.2, alpha=0.6)
        ax.set_xticks(x_base)
        ax.set_xticklabels(all_classes, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _candidate_checkpoints(feature: str) -> list[tuple[str, Path]]:
    MODEL_DIR = MODEL_BASE / feature
    FINETUNE_DIR = MODEL_DIR / "finetune"
    pairs = []
    if (FINETUNE_DIR / "best_model_finetuned.pt").exists():
        pairs.append(("Finetuned best", FINETUNE_DIR / "best_model_finetuned.pt"))
    if (FINETUNE_DIR / "latest_checkpoint_finetune.pt").exists():
        pairs.append(("Finetuned latest", FINETUNE_DIR / "latest_checkpoint_finetune.pt"))
    if (MODEL_DIR / "best_model.pt").exists():
        pairs.append(("Stage-1 best", MODEL_DIR / "best_model.pt"))
    if (MODEL_DIR / "latest_checkpoint.pt").exists():
        pairs.append(("Stage-1 latest", MODEL_DIR / "latest_checkpoint.pt"))
    if not pairs:
        sys.exit(f"No checkpoints found for feature '{feature}'. Run train_single.py --feature {feature} first.")
    return pairs


def evaluate_checkpoint(ckpt_path: Path, loader: DataLoader, device: torch.device, feature: str) -> dict:
    """Single-head evaluation: one task only."""
    model, _, id2label, _ = load_model(feature, ckpt_path, device)
    counts = {"correct": defaultdict(int), "total": defaultdict(int)}
    overall = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="  batches", leave=False, unit="batch"):
            logits = model(batch["input_values"].to(device))
            labels = batch[feature].to(device)
            preds = logits.argmax(1)
            mask = (preds == labels)
            overall += mask.sum().item()
            for gt_idx, ok in zip(labels.cpu().tolist(), mask.cpu().tolist()):
                lbl = id2label[gt_idx]
                counts["total"][lbl] += 1
                counts["correct"][lbl] += int(ok)
            total += labels.size(0)
    n = max(total, 1)
    per_class = {
        lbl: counts["correct"][lbl] / max(counts["total"][lbl], 1)
        for lbl in sorted(counts["total"])
    }
    return {
        "n": total,
        "accuracy": {feature: overall / n},
        "per_class": {feature: per_class},
        "support": {feature: dict(counts["total"])},
    }


def _bar_str(acc: float, width: int = 20) -> str:
    filled = round(acc * width)
    return "█" * filled + "░" * (width - filled)


def print_report(label: str, split: str, ckpt_path: Path, res: dict, feature: str) -> None:
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  {label}  |  split: {split}  |  n={res['n']}  |  feature: {feature}")
    print(f"  Checkpoint: {ckpt_path}")
    print(sep)
    acc = res["accuracy"][feature]
    print(f"\n  {feature.upper():<10}  overall: {acc*100:6.2f}%  {_bar_str(acc)}")
    for cls in sorted(res["per_class"][feature]):
        ca = res["per_class"][feature][cls]
        ns = res["support"][feature].get(cls, 0)
        print(f"    {cls:<18}  {ca*100:5.1f}%  n={ns:>4}  {_bar_str(ca, 12)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate single-head HuBERT.")
    parser.add_argument("--feature", default="emotion", choices=list(FEATURES), help="Feature to evaluate.")
    parser.add_argument("--split", default="both", choices=["val", "test", "train", "both", "all"],
                        help="Split(s) to evaluate. both=val+test, all=train+val+test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    feature = args.feature

    MODEL_DIR = MODEL_BASE / feature
    FINETUNE_DIR = MODEL_DIR / "finetune"
    S1_METRICS = MODEL_DIR / "training_metrics.json"
    FT_METRICS = FINETUNE_DIR / "training_metrics_finetune.json"

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Feature: {feature}")
    print(f"Device:  {device}")
    print(f"Outputs: {EVAL_DIR}/\n")

    print("Generating training curve plots…")
    s1_metrics = _load_json(S1_METRICS)
    ft_metrics = _load_json(FT_METRICS)
    if not s1_metrics and not ft_metrics:
        print("  WARNING: no training metrics — skipping curve plots.")
    else:
        plot_training_curves(s1_metrics, ft_metrics, feature, EVAL_DIR / f"training_curves_{feature}.png")
        plot_layer_weights(s1_metrics, ft_metrics, feature, EVAL_DIR / f"layer_weights_{feature}.png")

    print("\nLoading dataset…")
    dataset = load()
    emotion_encoder, gender_encoder, age_encoder = build_label_encoders(dataset)

    def _cast(name: str):
        return dataset[name].cast_column("audio", Audio(decode=False))

    if args.split in ("both", "all"):
        want = (["train"] if args.split == "all" else []) + ["validation", "val", "test"]
        splits_to_eval = {}
        for c in want:
            if c in dataset and c not in splits_to_eval:
                splits_to_eval[c] = _cast(c)
        if not splits_to_eval:
            splits_to_eval = {list(dataset.keys())[-1]: _cast(list(dataset.keys())[-1])}
    elif args.split in dataset:
        splits_to_eval = {args.split: _cast(args.split)}
    else:
        sys.exit(f"Split '{args.split}' not in dataset. Available: {list(dataset.keys())}")

    print("Evaluating on splits:", list(splits_to_eval.keys()))

    from transformers import Wav2Vec2FeatureExtractor
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

    checkpoints = _candidate_checkpoints(feature)
    evaluated = []
    all_results = defaultdict(dict)

    for label, ckpt_path in checkpoints:
        print(f"\n{'─'*56}")
        print(f"  Checkpoint: {label}")
        print(f"  Path: {ckpt_path}")
        for split_name, split_data in splits_to_eval.items():
            print(f"\n  Evaluating split: '{split_name}'  ({len(split_data)} rows)")
            ds = AudioDataset(split_data, processor, emotion_encoder, gender_encoder, age_encoder)
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            )
            res = evaluate_checkpoint(ckpt_path, loader, device, feature)
            print_report(label, split_name, ckpt_path, res, feature)
            evaluated.append((label, split_name, ckpt_path, res))
            all_results[label][split_name] = res

    json_path = EVAL_DIR / f"eval_results_{feature}.json"
    json_out = {}
    for label, split_name, ckpt_path, res in evaluated:
        entry = json_out.setdefault(label, {"checkpoint": str(ckpt_path)})
        entry[split_name] = res
    json_path.write_text(json.dumps(json_out, indent=2))
    print(f"Results saved: {json_path}")

    print("\nGenerating accuracy bar charts…")
    plot_accuracy_bars(dict(all_results), feature, EVAL_DIR / f"accuracy_bars_{feature}.png")

    print(f"\nDone. All outputs in {EVAL_DIR}/")


if __name__ == "__main__":
    main()
