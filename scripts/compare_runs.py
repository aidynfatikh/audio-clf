#!/usr/bin/env python3
"""Compare all runs (01, 02, 03) and latest eval using only JSONs inside run dirs.

Expects:
  - 01_hold_lr_run/, 02_warmup_lr_run/, 03_regularized_run/ (each optional)
    - eval/eval_results.json  or  eval_results.json
    - models/training_metrics.json  (optional, for curves)
    - models/finetune/training_metrics_finetune.json  (optional)
  - eval/eval_results.json  (current/latest run)

Outputs (in eval/run_comparison/):
  - test_accuracy_bars.png   — test acc by run and task (grouped bars)
  - val_vs_test.png         — val vs test acc per run/task (gap view)
  - training_curves.png     — val loss & val acc over epochs, all runs overlaid
  - run_comparison.json     — extracted accuracies for reference
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_NAMES = ["01_hold_lr_run", "02_warmup_lr_run", "03_regularized_run"]
EVAL_DIR = REPO_ROOT / "eval"
OUT_DIR = EVAL_DIR / "run_comparison"
TASKS = ("emotion", "gender", "age")
SPLITS = ("validation", "test")
CHECKPOINT_KEY = "Finetuned best"  # primary checkpoint to compare across runs


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def discover_run_evals() -> dict[str, dict]:
    """Collect eval_results.json per run. Keys: run_name -> full eval_results dict."""
    out = {}
    for name in RUN_NAMES:
        run_dir = REPO_ROOT / name
        if not run_dir.is_dir():
            continue
        for candidate in [run_dir / "eval" / "eval_results.json", run_dir / "eval_results.json"]:
            data = _load_json(candidate)
            if data is not None and isinstance(data, dict):
                out[name] = data
                break
    # Latest run: current eval/
    latest = _load_json(EVAL_DIR / "eval_results.json")
    if latest is not None and isinstance(latest, dict):
        out["eval"] = latest
    return out


def discover_training_metrics() -> dict[str, tuple[list, list]]:
    """Per run: (s1_metrics, ft_metrics) from models/ and models/finetune/."""
    out = {}
    for name in RUN_NAMES:
        run_dir = REPO_ROOT / name
        if not run_dir.is_dir():
            continue
        s1 = _load_json(run_dir / "models" / "training_metrics.json")
        ft = _load_json(run_dir / "models" / "finetune" / "training_metrics_finetune.json")
        if s1 is not None and not isinstance(s1, list):
            s1 = None
        if ft is not None and not isinstance(ft, list):
            ft = None
        out[name] = (s1 or [], ft or [])
    # Current run: repo models/
    s1 = _load_json(REPO_ROOT / "models" / "training_metrics.json")
    ft = _load_json(REPO_ROOT / "models" / "finetune" / "training_metrics_finetune.json")
    if s1 is not None and not isinstance(s1, list):
        s1 = None
    if ft is not None and not isinstance(ft, list):
        ft = None
    out["eval"] = (s1 or [], ft or [])
    return out


def extract_accuracies(run_evals: dict[str, dict]) -> dict[str, dict]:
    """From run_evals, build {run_name: {checkpoint: {split: {task: acc}}}}."""
    result = {}
    for run_name, eval_data in run_evals.items():
        result[run_name] = {}
        for ckpt_label, ckpt_data in eval_data.items():
            if ckpt_label == "checkpoint" or not isinstance(ckpt_data, dict):
                continue
            accs = {}
            for split in SPLITS:
                if split not in ckpt_data or not isinstance(ckpt_data[split], dict):
                    continue
                acc = ckpt_data[split].get("accuracy")
                if isinstance(acc, dict):
                    accs[split] = {t: acc.get(t, float("nan")) for t in TASKS}
            if accs:
                result[run_name][ckpt_label] = accs
    return result


def plot_test_accuracy_bars(extracted: dict[str, dict], out_path: Path) -> None:
    """Grouped bar: runs x tasks, using primary checkpoint per run."""
    runs = []
    for r in RUN_NAMES + ["eval"]:
        if r not in extracted:
            continue
        ckpts = extracted[r]
        # Prefer "Finetuned best", else first checkpoint that has test
        accs = None
        for key in [CHECKPOINT_KEY, "Finetuned best", "Finetuned latest"]:
            if key in ckpts and "test" in ckpts[key]:
                accs = ckpts[key]["test"]
                break
        if accs is None and ckpts:
            first = next(iter(ckpts.values()))
            accs = first.get("test", first.get("validation", {}))
        if accs:
            runs.append((r, accs))

    if not runs:
        print("  No data for test accuracy bars — skipping.")
        return

    run_labels, _ = zip(*runs)
    x = np.arange(len(run_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, task in enumerate(TASKS):
        vals = [acc.get(task, float("nan")) for _, acc in runs]
        vals = [v if not np.isnan(v) else 0.0 for v in vals]
        ax.bar(x + i * width - width, vals, width, label=task.capitalize())

    ax.set_ylabel("Test accuracy")
    ax.set_title("Test accuracy by run (Finetuned best)")
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_val_vs_test(extracted: dict[str, dict], out_path: Path) -> None:
    """Val vs test accuracy per run/task: one subplot per task, grouped bar (val / test) per run."""
    runs = []
    for r in RUN_NAMES + ["eval"]:
        if r not in extracted:
            continue
        ckpts = extracted[r]
        accs = None
        for key in [CHECKPOINT_KEY, "Finetuned best", "Finetuned latest"]:
            if key in ckpts and "test" in ckpts[key] and "validation" in ckpts[key]:
                accs = (ckpts[key]["validation"], ckpts[key]["test"])
                break
        if accs is None and ckpts:
            first = next(iter(ckpts.values()))
            accs = (first.get("validation", {}), first.get("test", {}))
        if accs and accs[0] and accs[1]:
            runs.append((r, accs[0], accs[1]))

    if not runs:
        print("  No data for val vs test — skipping.")
        return

    run_labels = [r[0] for r in runs]
    n_runs = len(run_labels)
    x = np.arange(n_runs)
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, task in zip(axes, TASKS):
        val_acc = [r[1].get(task, float("nan")) for r in runs]
        test_acc = [r[2].get(task, float("nan")) for r in runs]
        val_acc = [v if not np.isnan(v) else 0.0 for v in val_acc]
        test_acc = [t if not np.isnan(t) else 0.0 for t in test_acc]
        ax.bar(x - width / 2, val_acc, width, label="Val", color="#4C72B0", alpha=0.9)
        ax.bar(x + width / 2, test_acc, width, label="Test", color="#DD8452", alpha=0.9)
        ax.set_ylabel("Accuracy")
        ax.set_title(task.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Validation vs Test accuracy by run (Finetuned best)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _series(epochs: list[dict], key_path: list[str]) -> list[float]:
    out = []
    for e in epochs:
        node = e
        for k in key_path:
            node = node.get(k, {}) if isinstance(node, dict) else {}
        if isinstance(node, (int, float)):
            out.append(float(node))
        else:
            out.append(float("nan"))
    return out


def plot_training_curves(metrics: dict[str, tuple[list, list]], out_path: Path) -> None:
    """Overlay val total loss and val emotion_acc over epochs for each run."""
    colours = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for (run_name, (s1, ft)), color in zip(metrics.items(), colours):
        if not s1 and not ft:
            continue
        n_s1 = len(s1)
        x_s1 = list(range(1, n_s1 + 1))
        x_ft = list(range(n_s1 + 1, n_s1 + len(ft) + 1))
        x_all = x_s1 + x_ft
        loss_s1 = _series(s1, ["val", "total"])
        loss_ft = _series(ft, ["val", "total"])
        loss_all = loss_s1 + loss_ft
        acc_s1 = _series(s1, ["val", "emotion_acc"])
        acc_ft = _series(ft, ["val", "emotion_acc"])
        acc_all = acc_s1 + acc_ft
        ax_loss.plot(x_all, loss_all, "-", color=color, lw=2, label=run_name)
        ax_acc.plot(x_all, acc_all, "-", color=color, lw=2, label=run_name)

    ax_loss.set_ylabel("Val loss (total)")
    ax_loss.set_title("Validation total loss — all runs")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)
    ax_acc.set_ylabel("Val accuracy (emotion)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_title("Validation emotion accuracy — all runs")
    ax_acc.legend(fontsize=8)
    ax_acc.set_ylim(0, 1.05)
    ax_acc.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Run comparison (JSON-only)")
    print("Output dir:", OUT_DIR)

    run_evals = discover_run_evals()
    if not run_evals:
        print("No eval_results.json found in any run dir or eval/. Exiting.")
        return
    print("Runs with eval JSONs:", list(run_evals.keys()))

    extracted = extract_accuracies(run_evals)
    with open(OUT_DIR / "run_comparison.json", "w") as f:
        json.dump(extracted, f, indent=2)
    print("  Saved: run_comparison.json")

    plot_test_accuracy_bars(extracted, OUT_DIR / "test_accuracy_bars.png")
    plot_val_vs_test(extracted, OUT_DIR / "val_vs_test.png")

    metrics = discover_training_metrics()
    if any(s1 or ft for s1, ft in metrics.values()):
        plot_training_curves(metrics, OUT_DIR / "training_curves.png")
    else:
        print("  No training_metrics found — skipping training curves.")

    print("Done.")


if __name__ == "__main__":
    main()
