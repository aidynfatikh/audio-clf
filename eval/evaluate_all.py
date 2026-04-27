#!/usr/bin/env python3
"""Evaluate every stage-2 checkpoint under results/results/exp* on a single
combined test split.

For each (run, corpus) pair we report:
    n, emotion_acc, gender_acc, age_acc, mean_loss

All checkpoints score the SAME rows so numbers are directly comparable.

Usage
-----
    # build the canonical combined test split once
    python3 scripts/build_splits.py --config configs/splits/combined_eval_v1.yaml

    # then score every fine-tuned checkpoint against it
    python3 eval/evaluate_all.py \
        --split-dir splits/combined_eval_v1 \
        --runs-glob 'results/results/exp*/finetune/best_model_finetuned.pt' \
        --out results/combined_test_metrics

Outputs
-------
    <out>.csv            — flat rows: run, corpus, n, emo_acc, gen_acc, age_acc, loss
    <out>.md             — leaderboard pivot per task

Notes
-----
* Each run's ``label_encoders.json`` lives at ``<run_dir>/label_encoders.json``
  (one level up from the ``finetune/`` folder for stage-2 checkpoints, or in
  the run dir itself for stage-1). We resolve it relative to the checkpoint.
* Audio decode + processor/model loading reuse ``eval.inference.load_model``
  via a thin override so we don't depend on ``MODEL_DIR``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DATASETS_AUDIO_BACKEND", "soundfile")
os.environ.setdefault("TORCHCODEC_QUIET", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from multihead.model import MultiTaskBackbone
from multihead.utils import SAMPLE_RATE
from utils.config import build_feature_extractor
from utils.data import AudioDataset
from splits.materialize import materialize_named_test


TASKS = ("emotion", "gender", "age")


def _resolve_encoders(ckpt_path: Path) -> Path:
    """Find label_encoders.json near the checkpoint."""
    for parent in (ckpt_path.parent, ckpt_path.parent.parent, ckpt_path.parent.parent.parent):
        cand = parent / "label_encoders.json"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"label_encoders.json not found near {ckpt_path}")


def _load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    enc_path = _resolve_encoders(ckpt_path)
    with open(enc_path) as f:
        encoders = json.load(f)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone = ckpt.get("backbone_name", "hubert_base")
    pretrained = ckpt.get("pretrained", "facebook/hubert-base-ls960")
    model = MultiTaskBackbone(
        num_emotions=ckpt["num_emotions"],
        num_genders=ckpt["num_genders"],
        num_ages=ckpt["num_ages"],
        freeze_backbone=True,
        backbone_name=backbone,
        pretrained=pretrained,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device).eval()
    processor = build_feature_extractor(pretrained)
    return model, processor, encoders


@torch.no_grad()
def _score_one(model, loader, device) -> dict:
    correct = {t: 0 for t in TASKS}
    seen = {t: 0 for t in TASKS}
    loss_sum = 0.0
    loss_n = 0
    ce = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
    for batch in loader:
        x = batch["input_values"].to(device, non_blocking=True)
        il = batch.get("input_length")
        if il is not None:
            il = il.to(device, non_blocking=True)
        emo_l, gen_l, age_l = model(x, input_lengths=il)
        for t, logits in zip(TASKS, (emo_l, gen_l, age_l)):
            y = batch[t].to(device, non_blocking=True)
            mask = y != -100
            n = int(mask.sum().item())
            if n == 0:
                continue
            preds = logits.argmax(dim=1)
            correct[t] += int((preds[mask] == y[mask]).sum().item())
            seen[t] += n
            loss_sum += float(ce(logits, y).item())
            loss_n += n
    out = {"n_rows": len(loader.dataset)}
    for t in TASKS:
        out[f"{t}_acc"] = (correct[t] / seen[t]) if seen[t] else None
        out[f"{t}_n"] = seen[t]
    out["mean_loss"] = (loss_sum / loss_n) if loss_n else None
    return out


def evaluate_all(
    split_dir: Path,
    runs_glob: str,
    out_stem: Path,
    batch_size: int = 16,
    num_workers: int = 4,
    device: torch.device | None = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Split:  {split_dir}")

    print("Materialising per-corpus test slices…")
    per_corpus = materialize_named_test(split_dir)
    print(f"Corpora: {sorted(per_corpus.keys())}  sizes: " +
          ", ".join(f"{k}={len(v)}" for k, v in per_corpus.items()))

    runs = sorted(Path().glob(runs_glob))
    if not runs:
        sys.exit(f"No checkpoints matched: {runs_glob}")
    print(f"\nFound {len(runs)} checkpoints:")
    for r in runs:
        print(f"  · {r}")

    rows: list[dict] = []
    for ckpt in runs:
        run_id = ckpt.parent.parent.name if ckpt.parent.name == "finetune" else ckpt.parent.name
        stage = "stage2" if ckpt.parent.name == "finetune" else "stage1"
        print(f"\n=== {run_id} ({stage}) ===")
        try:
            model, processor, encoders = _load_model_from_ckpt(ckpt, device)
        except Exception as e:
            print(f"  load failed: {e}")
            continue

        for corpus, ds in per_corpus.items():
            tds = AudioDataset(
                ds, processor,
                encoders["emotion"], encoders["gender"], encoders["age"],
                is_train=False,
            )
            loader = DataLoader(tds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=(device.type == "cuda"))
            res = _score_one(model, loader, device)
            row = {"run": run_id, "stage": stage, "corpus": corpus, **res}
            rows.append(row)
            acc_str = " ".join(
                f"{t}={res[f'{t}_acc']:.3f}" if res[f"{t}_acc"] is not None else f"{t}=-"
                for t in TASKS
            )
            print(f"  {corpus:>10s} n={res['n_rows']:5d} | {acc_str} | loss={res['mean_loss']:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── write CSV ────────────────────────────────────────────────────────────
    out_csv = out_stem.with_suffix(".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "stage", "corpus", "n_rows",
              "emotion_acc", "emotion_n",
              "gender_acc",  "gender_n",
              "age_acc",     "age_n",
              "mean_loss"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"\nWrote {out_csv}")

    # ── write Markdown pivot ─────────────────────────────────────────────────
    out_md = out_stem.with_suffix(".md")
    corpora = sorted({r["corpus"] for r in rows})
    runs_seen = sorted({r["run"] for r in rows})
    lines = [f"# Combined-eval test metrics (`{split_dir.name}`)\n"]
    for task in TASKS:
        key = f"{task}_acc"
        lines.append(f"\n## {task} accuracy by run × corpus\n")
        header = "| run | " + " | ".join(corpora) + " | mean |"
        sep    = "|---|" + "|".join("---:" for _ in corpora) + "|---:|"
        lines += [header, sep]
        for run in runs_seen:
            cells = []
            vals = []
            for c in corpora:
                v = next((r[key] for r in rows if r["run"] == run and r["corpus"] == c), None)
                if v is None:
                    cells.append("—")
                else:
                    cells.append(f"{v:.3f}")
                    vals.append(v)
            mean_str = f"{sum(vals)/len(vals):.3f}" if vals else "—"
            lines.append(f"| `{run}` | " + " | ".join(cells) + f" | **{mean_str}** |")
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", required=True, type=Path,
                   help="splits/<name> dir (must already be built)")
    p.add_argument("--runs-glob", default="results/results/exp*/finetune/best_model_finetuned.pt",
                   help="Glob (relative to repo root) of checkpoints to evaluate")
    p.add_argument("--out", default="results/combined_test_metrics", type=Path,
                   help="Output stem; .csv and .md are written")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = p.parse_args()

    device = torch.device(args.device) if args.device else None
    evaluate_all(
        split_dir=args.split_dir,
        runs_glob=args.runs_glob,
        out_stem=args.out,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )


if __name__ == "__main__":
    main()
