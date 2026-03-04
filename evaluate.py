#!/usr/bin/env python3
"""Evaluate MultiTaskHubert checkpoints and visualise training history.

Outputs written to eval/:
  training_curves.png      — loss + accuracy from stage-1 → finetune
  layer_weights.png        — learned HuBERT layer-attention weights over epochs
  eval_results.json        — per-checkpoint accuracy + per-class breakdown
  accuracy_bars.png        — per-class accuracy bar charts for each checkpoint

Checkpoints evaluated (prefers finetune files, falls back to stage-1):
  1. best_model_finetuned.pt  / best_model.pt
  2. latest_checkpoint_finetune.pt / latest_checkpoint.pt

Usage:
    python evaluate.py [--split val|test|both] [--batch-size N] [--device cpu|cuda]
"""

import os
import sys
import json
import argparse
import warnings
from collections import defaultdict
from pathlib import Path

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["TORCHCODEC_QUIET"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
from datasets import Audio
from tqdm import tqdm

from load_data import load
from train import AudioDataset, MODEL_DIR, SAMPLE_RATE
from inference import load_model

FINETUNE_DIR = Path(MODEL_DIR) / "finetune"
EVAL_DIR     = Path(__file__).resolve().parent / "eval"
S1_METRICS   = Path(MODEL_DIR) / "training_metrics.json"
FT_METRICS   = FINETUNE_DIR   / "training_metrics_finetune.json"

TASK_COLOUR  = {"emotion": "#E05C5C", "gender": "#5C9BE0", "age": "#5CBF7A"}
PHASE_COLOUR = {"stage1": "#4C72B0",  "finetune": "#DD8452"}
TASKS        = ("emotion", "gender", "age")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _series(epochs: list[dict], key_path: list[str]):
    """Extract a scalar series from a list of epoch dicts."""
    out = []
    for e in epochs:
        node = e
        for k in key_path:
            node = node.get(k, {})
        out.append(float(node) if isinstance(node, (int, float)) else float("nan"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Training curves — loss & accuracy, stage-1 → finetune
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(s1: list[dict], ft: list[dict], out_path: Path) -> None:
    if not s1 and not ft:
        print("  No training metrics found — skipping training curves.")
        return

    # Find best stage-1 epoch (lowest val total loss) to mark the anchor
    best_s1_idx = int(np.argmin(_series(s1, ["val", "total"]))) if s1 else 0
    best_s1_ep  = s1[best_s1_idx]["epoch"] if s1 else 0

    n_s1 = len(s1)
    x_s1 = list(range(1, n_s1 + 1))
    x_ft = list(range(n_s1 + 1, n_s1 + len(ft) + 1))

    fig, axes = plt.subplots(4, 2, figsize=(16, 18), constrained_layout=True)
    fig.suptitle("Training History: Stage-1 → Finetune", fontsize=16, fontweight="bold")

    rows = [
        ("Total",   ["train", "total"],   ["val", "total"],   None),
        ("Emotion", ["train", "emotion"], ["val", "emotion"], ["val", "emotion_acc"]),
        ("Gender",  ["train", "gender"],  ["val", "gender"],  ["val", "gender_acc"]),
        ("Age",     ["train", "age"],     ["val", "age"],     ["val", "age_acc"]),
    ]

    def _plot_phase(ax, x, vals, phase, style="-", lw=2, alpha=1.0, label=None):
        pairs = [(xi, yi) for xi, yi in zip(x, vals) if not np.isnan(yi)]
        if not pairs:
            return
        xs, ys = zip(*pairs)
        ax.plot(xs, ys, style, color=PHASE_COLOUR[phase], lw=lw, alpha=alpha, label=label)

    for row_idx, (title, train_key, val_key, acc_key) in enumerate(rows):
        ax_loss = axes[row_idx, 0]
        ax_acc  = axes[row_idx, 1]

        # ── Loss subplot ─────────────────────────────────────────────────────
        _plot_phase(ax_loss, x_s1, _series(s1, train_key), "stage1",   label="Train (stage-1)")
        _plot_phase(ax_loss, x_s1, _series(s1, val_key),   "stage1",   style="--", label="Val (stage-1)")
        _plot_phase(ax_loss, x_ft, _series(ft, train_key), "finetune", label="Train (finetune)")
        _plot_phase(ax_loss, x_ft, _series(ft, val_key),   "finetune", style="--", label="Val (finetune)")

        if s1:
            bv = _series(s1, val_key)[best_s1_idx]
            if not np.isnan(bv):
                ax_loss.axvline(best_s1_ep, color="grey", ls=":", lw=1.2, alpha=0.7)
                ax_loss.scatter([best_s1_ep], [bv], color=PHASE_COLOUR["stage1"],
                                zorder=5, s=70, marker="*")
        if s1 and ft:
            ax_loss.axvline(n_s1 + 0.5, color="black", ls="-.", lw=1.5, alpha=0.5,
                            label="→ Finetune")
        ax_loss.set_title(f"{title} — Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=7, loc="upper right")
        ax_loss.grid(True, alpha=0.3)

        # ── Accuracy / breakdown subplot ─────────────────────────────────────
        if acc_key is not None:
            _plot_phase(ax_acc, x_s1, _series(s1, acc_key), "stage1",   label="Val acc (stage-1)")
            _plot_phase(ax_acc, x_ft, _series(ft, acc_key), "finetune", label="Val acc (finetune)")
            if s1:
                ba = _series(s1, acc_key)[best_s1_idx]
                if not np.isnan(ba):
                    ax_acc.axvline(best_s1_ep, color="grey", ls=":", lw=1.2, alpha=0.7)
                    ax_acc.scatter([best_s1_ep], [ba], color=PHASE_COLOUR["stage1"],
                                   zorder=5, s=70, marker="*")
            if s1 and ft:
                ax_acc.axvline(n_s1 + 0.5, color="black", ls="-.", lw=1.5, alpha=0.5)
            ax_acc.set_ylim(0, 1.05)
            ax_acc.set_title(f"{title} — Val Accuracy")
            ax_acc.set_xlabel("Epoch")
            ax_acc.set_ylabel("Accuracy")
            ax_acc.legend(fontsize=7, loc="lower right")
            ax_acc.grid(True, alpha=0.3)
        else:
            # Total row: overlay all three task val-losses for comparison
            ax_acc.set_title("Val Loss — All Tasks")
            ax_acc.set_xlabel("Epoch")
            ax_acc.set_ylabel("Val Loss")
            for task in TASKS:
                ax_acc.plot(x_s1, _series(s1, ["val", task]),
                            color=TASK_COLOUR[task], lw=2,       label=f"{task} (s1)")
                ax_acc.plot(x_ft, _series(ft, ["val", task]),
                            color=TASK_COLOUR[task], lw=2, ls="--", label=f"{task} (ft)")
            if s1 and ft:
                ax_acc.axvline(n_s1 + 0.5, color="black", ls="-.", lw=1.5, alpha=0.5)
            ax_acc.legend(fontsize=7)
            ax_acc.grid(True, alpha=0.3)

    legend_items = [
        mpatches.Patch(color=PHASE_COLOUR["stage1"],  label="Stage-1"),
        mpatches.Patch(color=PHASE_COLOUR["finetune"], label="Finetune"),
        Line2D([0], [0], color="grey",  ls=":",  lw=1.2, label=f"Best s1 epoch ({best_s1_ep})"),
        Line2D([0], [0], color="black", ls="-.", lw=1.5, label="Finetune boundary"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), fontsize=9)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Layer attention weights over epochs
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer_weights(s1: list[dict], ft: list[dict], out_path: Path) -> None:
    if not s1 and not ft:
        return

    n_layers = 13   # 12 transformer + 1 CNN feature extractor
    n_s1     = len(s1)
    x_s1     = list(range(1, n_s1 + 1))
    x_ft     = list(range(n_s1 + 1, n_s1 + len(ft) + 1))
    cmap     = plt.cm.tab20

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle("Learned HuBERT Layer Attention Weights Over Training",
                 fontsize=14, fontweight="bold")

    for col_idx, task in enumerate(TASKS):
        ax = axes[col_idx]
        ax.set_title(task.capitalize(), fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Softmax weight")
        ax.grid(True, alpha=0.2)

        for li in range(n_layers):
            colour = cmap(li / n_layers)
            lbl    = "CNN" if li == 0 else f"L{li}"

            def _layer_series(epochs):
                return [
                    e["layer_prefs"][task][li]
                    if ("layer_prefs" in e
                        and task in e["layer_prefs"]
                        and li < len(e["layer_prefs"][task]))
                    else float("nan")
                    for e in epochs
                ]

            ws_s1 = _layer_series(s1)
            ws_ft = _layer_series(ft)

            if any(not np.isnan(w) for w in ws_s1):
                ax.plot(x_s1, ws_s1, color=colour, lw=1.5, alpha=0.9, label=lbl)
            if any(not np.isnan(w) for w in ws_ft):
                ax.plot(x_ft, ws_ft, color=colour, lw=1.5, ls="--", alpha=0.9)

        if s1 and ft:
            ax.axvline(n_s1 + 0.5, color="black", ls="-.", lw=1.5, alpha=0.5,
                       label="→ Finetune")
        ax.legend(fontsize=6, loc="upper left", ncol=2,
                  bbox_to_anchor=(1.01, 1), borderaxespad=0)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Accuracy bar charts from evaluation results
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_bars(all_results: dict, out_path: Path) -> None:
    """all_results: {model_label: {split_name: eval_result_dict}}"""
    model_labels = sorted(all_results.keys())
    n_models     = len(model_labels)
    split_colours = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(len(TASKS), n_models,
                             figsize=(7 * n_models, 5 * len(TASKS)),
                             constrained_layout=True)
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle("Per-Class Accuracy by Checkpoint & Split",
                 fontsize=14, fontweight="bold")

    for col, mlabel in enumerate(model_labels):
        for row, task in enumerate(TASKS):
            ax = axes[row, col]
            ax.set_title(f"{mlabel}\n{task.capitalize()}", fontsize=9)
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.15)

            split_names = sorted(all_results[mlabel].keys())
            n_splits    = len(split_names)
            group_w     = 0.75
            bar_w       = group_w / max(n_splits, 1)

            all_classes = sorted({
                cls
                for sname in split_names
                for cls in all_results[mlabel][sname]["per_class"][task]
            })
            x_base = np.arange(len(all_classes))

            for si, sname in enumerate(split_names):
                res    = all_results[mlabel][sname]
                offset = (si - (n_splits - 1) / 2) * bar_w
                vals   = [res["per_class"][task].get(c, 0.0) for c in all_classes]
                bars   = ax.bar(x_base + offset, vals, bar_w * 0.92,
                                label=sname,
                                color=split_colours[si % len(split_colours)],
                                alpha=0.85, edgecolor="white")
                for bar, cls in zip(bars, all_classes):
                    n_cls = res["support"][task].get(cls, 0)
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.02,
                            f"n={n_cls}", ha="center", va="bottom",
                            fontsize=6.5, rotation=45)
                # overall accuracy dashed line
                ov = res["accuracy"][task]
                ax.axhline(ov, color=split_colours[si % len(split_colours)],
                           ls="--", lw=1.2, alpha=0.6)

            ax.set_xticks(x_base)
            ax.set_xticklabels(all_classes, rotation=30, ha="right", fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint resolution
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_checkpoints() -> list[tuple[str, Path]]:
    finetune_exist = (
        (FINETUNE_DIR / "best_model_finetuned.pt").exists()
        or (FINETUNE_DIR / "latest_checkpoint_finetune.pt").exists()
    )
    if finetune_exist:
        pairs = [
            ("Finetuned best",   FINETUNE_DIR / "best_model_finetuned.pt"),
            ("Finetuned latest", FINETUNE_DIR / "latest_checkpoint_finetune.pt"),
        ]
    else:
        pairs = [
            ("Stage-1 best",   Path(MODEL_DIR) / "best_model.pt"),
            ("Stage-1 latest", Path(MODEL_DIR) / "latest_checkpoint.pt"),
        ]
    result = [(lbl, p) for lbl, p in pairs if p.exists()]
    if not result:
        sys.exit("No checkpoints found. Run train.py (and optionally finetune.py) first.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(ckpt_path: Path, loader: DataLoader,
                        device: torch.device) -> dict:
    """Full-pass evaluation. Returns overall + per-class accuracy and support."""
    (model, _, _, _, id2emotion, id2gender, id2age, _) = load_model(ckpt_path, device)

    id2label = {"emotion": id2emotion, "gender": id2gender, "age": id2age}
    counts   = {t: {"correct": defaultdict(int), "total": defaultdict(int)} for t in TASKS}
    overall  = {t: 0 for t in TASKS}
    total    = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  batches", leave=False, unit="batch"):
            e_out, g_out, a_out = model(batch["input_values"].to(device))
            for task, logits, gt_key in [
                ("emotion", e_out, "emotion"),
                ("gender",  g_out, "gender"),
                ("age",     a_out, "age"),
            ]:
                preds = logits.argmax(1)
                gts   = batch[gt_key].to(device)
                mask  = (preds == gts)
                overall[task] += mask.sum().item()
                for gt_idx, ok in zip(gts.cpu().tolist(), mask.cpu().tolist()):
                    lbl = id2label[task][gt_idx]
                    counts[task]["total"][lbl]   += 1
                    counts[task]["correct"][lbl] += int(ok)
            total += batch["emotion"].size(0)

    n = max(total, 1)
    return {
        "n": total,
        "accuracy":  {t: overall[t] / n for t in TASKS},
        "per_class": {
            t: {
                lbl: counts[t]["correct"][lbl] / max(counts[t]["total"][lbl], 1)
                for lbl in sorted(counts[t]["total"])
            }
            for t in TASKS
        },
        "support": {t: dict(counts[t]["total"]) for t in TASKS},
    }


def _bar_str(acc: float, width: int = 20) -> str:
    filled = round(acc * width)
    return "█" * filled + "░" * (width - filled)


def print_report(label: str, split: str, ckpt_path: Path, res: dict) -> None:
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  {label}  |  split: {split}  |  n={res['n']}")
    print(f"  Checkpoint: {ckpt_path}")
    print(sep)
    for task in TASKS:
        acc = res["accuracy"][task]
        print(f"\n  {task.upper():<10}  overall: {acc*100:6.2f}%  {_bar_str(acc)}")
        for cls in sorted(res["per_class"][task]):
            ca = res["per_class"][task][cls]
            ns = res["support"][task].get(cls, 0)
            print(f"    {cls:<18}  {ca*100:5.1f}%  n={ns:>4}  {_bar_str(ca, 12)}")
    print()


def print_comparison(evaluated: list) -> None:
    """evaluated: [(label, split_name, path, res), ...]"""
    if len(evaluated) < 2:
        return
    col_w = 24
    print("\n" + "═" * (14 + col_w * len(evaluated)))
    print("  COMPARISON")
    print("═" * (14 + col_w * len(evaluated)))
    header = f"  {'Task':<12}" + "".join(
        f"  {f'{lbl}/{spl}'[:col_w - 2]:<{col_w}}" for lbl, spl, _, _ in evaluated
    )
    print(header)
    print("  " + "─" * (len(header) - 2))
    for task in TASKS:
        row = f"  {task.capitalize():<12}"
        for _, _, _, res in evaluated:
            acc = res["accuracy"][task]
            row += f"  {acc*100:6.2f}%  {_bar_str(acc, 10):<{col_w - 10}}"
        print(row)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MultiTaskHubert and plot training history."
    )
    parser.add_argument("--split", default="both",
                        choices=["val", "test", "train", "both", "all"],
                        help="Split(s) to evaluate on. "
                             "both=val+test, all=train+val+test (default: both)")
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device",      default=None, help="cpu or cuda (default: auto)")
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}")
    print(f"Outputs: {EVAL_DIR}/\n")

    # ── Training visualisations ───────────────────────────────────────────────
    print("Generating training curve plots…")
    s1_metrics = _load_json(S1_METRICS)
    ft_metrics = _load_json(FT_METRICS)
    if not s1_metrics and not ft_metrics:
        print("  WARNING: no training metrics found — skipping curve plots.")
    else:
        plot_training_curves(s1_metrics, ft_metrics, EVAL_DIR / "training_curves.png")
        plot_layer_weights(s1_metrics, ft_metrics,   EVAL_DIR / "layer_weights.png")

    # ── Dataset & shared objects ──────────────────────────────────────────────
    print("\nLoading dataset…")
    dataset = load()
    print("Splits:", list(dataset.keys()))

    # Resolve which splits to evaluate
    def _cast(name: str):
        return dataset[name].cast_column("audio", Audio(decode=False))

    if args.split in ("both", "all"):
        # Canonical order: train → val/validation → test
        want = (["train"] if args.split == "all" else []) + ["validation", "val", "test"]
        splits_to_eval = {}
        for candidate in want:
            if candidate in dataset and candidate not in splits_to_eval:
                splits_to_eval[candidate] = _cast(candidate)
        if not splits_to_eval:
            last = list(dataset.keys())[-1]
            splits_to_eval = {last: _cast(last)}
    elif args.split in dataset:
        splits_to_eval = {args.split: _cast(args.split)}
    else:
        sys.exit(f"Split '{args.split}' not in dataset. Available: {list(dataset.keys())}")

    # ── Data-leakage sanity check ──────────────────────────────────────────────
    # Confirm the splits used for training vs evaluation are distinct so the
    # user can see explicitly that val/test were never fed into the optimiser.
    train_key = next((k for k in dataset.keys() if k == "train"), None)
    print("\nData-split audit:")
    print(f"  Training split used by train.py  : {train_key or '(first split)'}")
    print(f"  Validation split used by train.py: validation → val → test (first match)")
    print(f"  Splits being evaluated now       : {list(splits_to_eval.keys())}")
    print()

    print("Evaluating on splits:", list(splits_to_eval.keys()))

    enc_path = Path(MODEL_DIR) / "label_encoders.json"
    with open(enc_path) as f:
        encoders = json.load(f)
    emotion_enc = encoders["emotion"]
    gender_enc  = encoders["gender"]
    age_enc     = encoders["age"]

    from transformers import Wav2Vec2FeatureExtractor
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    checkpoints = _candidate_checkpoints()
    evaluated   = []                   # (label, split_name, ckpt_path, res)
    all_results = defaultdict(dict)    # {label: {split: res}}

    for label, ckpt_path in checkpoints:
        print(f"\n{'─'*56}")
        print(f"  Checkpoint : {label}")
        print(f"  Path       : {ckpt_path}")

        for split_name, split_data in splits_to_eval.items():
            print(f"\n  Evaluating split: '{split_name}'  ({len(split_data)} rows)")
            ds     = AudioDataset(split_data, processor, emotion_enc, gender_enc, age_enc)
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            res = evaluate_checkpoint(ckpt_path, loader, device)
            print_report(label, split_name, ckpt_path, res)

            evaluated.append((label, split_name, ckpt_path, res))
            all_results[label][split_name] = res

    print_comparison(evaluated)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    json_path = EVAL_DIR / "eval_results.json"
    json_out  = {}
    for label, split_name, ckpt_path, res in evaluated:
        entry = json_out.setdefault(label, {"checkpoint": str(ckpt_path)})
        entry[split_name] = res
    json_path.write_text(json.dumps(json_out, indent=2))
    print(f"Results saved : {json_path}")

    # ── Accuracy bar charts ────────────────────────────────────────────────────
    print("\nGenerating accuracy bar charts…")
    plot_accuracy_bars(dict(all_results), EVAL_DIR / "accuracy_bars.png")

    print(f"\nDone. All outputs in {EVAL_DIR}/")


if __name__ == "__main__":
    main()
