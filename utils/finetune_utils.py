"""Stage-2 layer analysis, unfreezing, and discriminative learning-rate optimizer."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from multihead.model import MultiTaskHubert
from utils.misc import unwrap

# HuBERT has 13 hidden states: index 0 = feature-projection output,
# indices 1-12 = transformer encoder layers [0..11].
NUM_HUBERT_LAYERS = 13
NUM_TRANSFORMER_LAYERS = 12  # encoder.layers[0] … encoder.layers[11]


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
        return {t: [1.0 / NUM_HUBERT_LAYERS] * NUM_HUBERT_LAYERS for t in tasks}

    result = {}
    for task in tasks:
        total = sum(sums[task]) or 1.0
        result[task] = [v / total for v in sums[task]]
    return result


def latest_layer_prefs(metrics_dir: Path) -> dict[str, list[float]]:
    """
    Return the per-task layer preferences from the **last epoch** of the
    **most recently modified** training_metrics*.json file.
    """
    tasks = ["emotion", "gender", "age"]
    candidates = list(metrics_dir.glob("training_metrics*.json"))
    if not candidates:
        return aggregate_layer_importance(metrics_dir)

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    prefs = load_layer_prefs(newest)
    if prefs is None:
        return aggregate_layer_importance(metrics_dir)

    result = {}
    for task in tasks:
        raw = prefs.get(task, [1.0 / NUM_HUBERT_LAYERS] * NUM_HUBERT_LAYERS)
        total = sum(raw) or 1.0
        result[task] = [v / total for v in raw]
    return result


def rank_transformer_layers(layer_prefs: dict[str, list[float]]) -> list[int]:
    """
    Rank the 12 transformer encoder layers by aggregated importance across tasks.
    Returns encoder indices (0-based for model.hubert.encoder.layers[i]), best-first.
    """
    tasks = list(layer_prefs.keys())
    avg = []
    for enc_idx in range(NUM_TRANSFORMER_LAYERS):
        hidden_state_idx = enc_idx + 1
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
        print(
            f"  #{rank:2d}  encoder.layers[{enc_idx:2d}]  avg={avg_score:.4f}  "
            f"({', '.join(f'{t[0]}:{v:.3f}' for t, v in scores.items())})  {bar}"
        )

    print("\nRecommended --unfreeze_top_n choices:")
    for n in [2, 4, 6, 8]:
        layers = ranked[:n]
        print(f"  top {n}: encoder.layers{sorted(layers)}")
    print("=" * 60 + "\n")


def unfreeze_layers(
    model: MultiTaskHubert,
    transformer_layer_indices: list[int],
    unfreeze_feature_proj: bool = False,
) -> list[nn.Parameter]:
    """Unfreeze specific encoder layers (and optionally feature_projection)."""
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


def build_optimizer(
    model: MultiTaskHubert,
    layer_indices: list[int],
    backbone_lr_top: float,
    layer_decay: float,
    head_lr: float,
) -> optim.Optimizer:
    """Discriminative-LR AdamW for unfrozen layers + heads."""
    param_groups: list[dict] = []

    m = unwrap(model)

    for enc_idx in layer_indices:
        lr_n = backbone_lr_top * (layer_decay ** (NUM_TRANSFORMER_LAYERS - enc_idx))
        params = [p for p in m.hubert.encoder.layers[enc_idx].parameters() if p.requires_grad]
        if params:
            param_groups.append({
                "params": params,
                "lr": lr_n,
                "name": f"encoder.layers[{enc_idx}]",
            })

    fp_params = [p for p in m.hubert.feature_projection.parameters() if p.requires_grad]
    if fp_params:
        lr_fp = backbone_lr_top * (layer_decay ** (NUM_TRANSFORMER_LAYERS + 1))
        param_groups.append({"params": fp_params, "lr": lr_fp, "name": "feature_projection"})

    head_params = (
        list(m.emotion_head.parameters())
        + list(m.gender_head.parameters())
        + list(m.age_head.parameters())
        + [m.emotion_weights, m.gender_weights, m.age_weights]
    )
    param_groups.append({"params": head_params, "lr": head_lr, "name": "heads"})

    _fused = torch.cuda.is_available()
    try:
        return optim.AdamW(param_groups, weight_decay=0.01, fused=_fused)
    except (TypeError, RuntimeError):
        return optim.AdamW(param_groups, weight_decay=0.01)


def print_lr_schedule(layer_indices: list[int], backbone_lr_top: float, layer_decay: float, head_lr: float) -> None:
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


_print_lr_schedule = print_lr_schedule
