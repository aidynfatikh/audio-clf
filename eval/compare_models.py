#!/usr/bin/env python3
"""
Evaluate one or more model checkpoints against:
  1. The batch01 holdout set  (1 008 balanced, non-augmented samples)
  2. A KazEmoTTS test split   (10 000 samples, skipping the first 20 000 used in training)

Usage:
  python eval/compare_models.py  path/to/model1.pt  path/to/model2.pt  ...
  python eval/compare_models.py  path/to/model1.pt  --output results/comparison.json
  python eval/compare_models.py  path/to/model1.pt  --batch-size 16 --kazemo-n 5000
  python eval/compare_models.py  path/to/model1.pt  --no-kazemo

The label_encoders.json is resolved from the checkpoint directory (or its parent).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["TORCHCODEC_QUIET"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from datasets import Audio, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor

from eval.validation_holdout import (
    first_index_by_row_id,
    load_validation_sample_ids,
    val_indices_from_manifest,
)
from loaders.kazemo.load_data import load_kazemotts
from loaders.load_data import DATA_DIR, read_audio
from multihead.model import MultiTaskHubert
from multihead.utils import SAMPLE_RATE

# ── defaults ────────────────────────────────────────────────────────────────
MANIFEST_PATH = _REPO_ROOT / "results" / "validation_subset_manifest.json"
HF_BATCH01_ID = os.environ.get("HF_BATCH01_ID", "01gumano1d/batch01-validation-test")
HF_BATCH01_CACHE = _REPO_ROOT / "data" / "batch01-validation-test"
KAZEMO_TRAIN_N = 20_000   # samples used during training (skip these)
KAZEMO_EVAL_N = 10_000    # eval samples to load after the skip


# ── audio dataset ────────────────────────────────────────────────────────────

class _AudioRows(Dataset):
    """Minimal dataset over a list of row-dicts for inference."""

    def __init__(
        self,
        rows: list[dict],
        processor: Wav2Vec2FeatureExtractor,
        emotion_encoder: dict[str, int] | None,
        gender_encoder: dict[str, int] | None,
        age_encoder: dict[str, int] | None,
        max_length: int = 160_000,
    ):
        self.rows = rows
        self.processor = processor
        self.emotion_encoder = emotion_encoder or {}
        self.gender_encoder = gender_encoder or {}
        self.age_encoder = age_encoder or {}
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        audio_data, sr = read_audio(row["audio"])

        if sr != SAMPLE_RATE:
            import librosa
            audio_data = librosa.resample(
                audio_data.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE
            )
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32, copy=False)
        if len(audio_data) > self.max_length:
            audio_data = audio_data[: self.max_length]
        else:
            audio_data = np.pad(audio_data, (0, self.max_length - len(audio_data)))

        inputs = self.processor(
            audio_data,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        item: dict = {
            "input_values": inputs.input_values.squeeze(0),
        }
        attn = getattr(inputs, "attention_mask", None)
        item["attention_mask"] = (
            attn.squeeze(0) if attn is not None
            else torch.ones(item["input_values"].shape, dtype=torch.long)
        )

        emo = row.get("emotion")
        item["emotion"] = torch.tensor(
            self.emotion_encoder.get(emo, -1) if emo else -1, dtype=torch.long
        )
        gen = row.get("gender")
        item["gender"] = torch.tensor(
            self.gender_encoder.get(gen, -1) if gen else -1, dtype=torch.long
        )
        age = row.get("age_category")
        item["age"] = torch.tensor(
            self.age_encoder.get(age, -1) if age else -1, dtype=torch.long
        )
        return item


# ── metrics ──────────────────────────────────────────────────────────────────

def _compute_metrics(
    preds: list[int],
    labels: list[int],
    id2label: dict[int, str],
) -> dict:
    """Accuracy + per-class accuracy + macro-F1 for one task."""
    classes = sorted(id2label.keys())
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    correct = total = 0
    for p, g in zip(preds, labels):
        if g == -1:
            continue
        total += 1
        if p == g:
            correct += 1
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    per_class: dict[str, float] = {}
    f1_scores: list[float] = []
    for c in classes:
        name = id2label[c]
        support = tp[c] + fn[c]
        per_class[name] = round(tp[c] / support, 6) if support else 0.0
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1_scores.append(f1)

    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_f1": round(float(np.mean(f1_scores)), 6) if f1_scores else 0.0,
        "per_class_accuracy": per_class,
        "n": total,
    }


def _run_eval(
    model: torch.nn.Module,
    rows: list[dict],
    processor: Wav2Vec2FeatureExtractor,
    emotion_encoder: dict[str, int],
    gender_encoder: dict[str, int],
    age_encoder: dict[str, int],
    device: torch.device,
    batch_size: int,
    tasks: set[str],
    desc: str = "Evaluating",
) -> dict:
    """Run inference over rows; return per-task metrics."""
    ds = _AudioRows(rows, processor, emotion_encoder, gender_encoder, age_encoder)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    id2emotion = {v: k for k, v in emotion_encoder.items()}
    id2gender  = {v: k for k, v in gender_encoder.items()}
    id2age     = {v: k for k, v in age_encoder.items()}

    all_preds:  dict[str, list[int]] = {"emotion": [], "gender": [], "age": []}
    all_labels: dict[str, list[int]] = {"emotion": [], "gender": [], "age": []}

    _use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if _use_bf16 else torch.float32

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            iv = batch["input_values"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                emo_logits, gen_logits, age_logits = model(iv, attention_mask=attn)
            all_preds["emotion"].extend(emo_logits.argmax(1).cpu().tolist())
            all_preds["gender"].extend(gen_logits.argmax(1).cpu().tolist())
            all_preds["age"].extend(age_logits.argmax(1).cpu().tolist())
            all_labels["emotion"].extend(batch["emotion"].tolist())
            all_labels["gender"].extend(batch["gender"].tolist())
            all_labels["age"].extend(batch["age"].tolist())

    results = {}
    if "emotion" in tasks:
        results["emotion"] = _compute_metrics(all_preds["emotion"], all_labels["emotion"], id2emotion)
    if "gender" in tasks:
        results["gender"]  = _compute_metrics(all_preds["gender"],  all_labels["gender"],  id2gender)
    if "age" in tasks:
        results["age"]     = _compute_metrics(all_preds["age"],     all_labels["age"],     id2age)
    return results


# ── data loading ─────────────────────────────────────────────────────────────

def _load_holdout_rows(manifest: Path, hf_id: str, cache: Path) -> list[dict]:
    sample_ids = load_validation_sample_ids(manifest)
    ds = load_dataset(hf_id, cache_dir=str(cache))
    split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    split = split.cast_column("audio", Audio(decode=False))
    id_to_idx = first_index_by_row_id(split)
    indices = val_indices_from_manifest(sample_ids, id_to_idx)
    rows = []
    for i in indices:
        row = dict(split[i])
        rows.append({
            "audio":        row["audio"],
            "emotion":      str(row.get("emotion", "") or "").strip() or None,
            "gender":       str(row.get("gender", "") or "").strip() or None,
            "age_category": str(row.get("age_category", "") or "").strip() or None,
        })
    return rows


def _load_kazemo_rows(cache_dir: str, n: int, skip: int) -> list[dict]:
    # Load skip+n samples then discard the first `skip` (training split).
    # This produces a non-overlapping eval split using the same deterministic
    # balanced order as the training loader.
    total = skip + n if n > 0 else None
    kz = load_kazemotts(cache_dir=cache_dir, max_samples=total)
    base = kz.get("train", kz[list(kz.keys())[0]])
    if skip > 0 and len(base) > skip:
        base = base.select(range(skip, len(base)))
    base = base.cast_column("audio", Audio(decode=False))
    rows = []
    for row in base:
        emo = str(row.get("emotion", "") or "").strip() or None
        if emo:
            rows.append({"audio": row["audio"], "emotion": emo,
                         "gender": None, "age_category": None})
    return rows


# ── checkpoint helpers ───────────────────────────────────────────────────────

def _resolve_encoders(ckpt_path: Path) -> Path:
    for candidate in (ckpt_path.parent, ckpt_path.parent.parent):
        p = candidate / "label_encoders.json"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"label_encoders.json not found next to {ckpt_path} or its parent. "
        "Make sure the model directory contains label_encoders.json."
    )


def _load_ckpt_meta(ckpt: dict, ckpt_path: Path) -> dict:
    meta: dict = {"path": str(ckpt_path)}
    for key in ("epoch", "global_step", "samples_seen", "val_loss",
                "best_val_loss", "num_emotions", "num_genders", "num_ages",
                "unfrozen_layers"):
        if key in ckpt:
            meta[key] = ckpt[key]
    # Infer stage from path / keys
    meta["stage"] = (
        "stage2_finetune" if "unfrozen_layers" in ckpt
        else "stage1_frozen"
    )
    return meta


# ── per-model evaluation ─────────────────────────────────────────────────────

def evaluate_model(
    ckpt_path: Path,
    holdout_rows: list[dict],
    kazemo_rows: list[dict] | None,
    device: torch.device,
    batch_size: int,
) -> dict:
    enc_path = _resolve_encoders(ckpt_path)
    with open(enc_path) as f:
        encoders = json.load(f)
    emotion_encoder: dict[str, int] = encoders["emotion"]
    gender_encoder:  dict[str, int] = encoders["gender"]
    age_encoder:     dict[str, int] = encoders["age"]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = _load_ckpt_meta(ckpt, ckpt_path)

    model = MultiTaskHubert(
        num_emotions=ckpt["num_emotions"],
        num_genders=ckpt["num_genders"],
        num_ages=ckpt["num_ages"],
        freeze_backbone=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

    print(f"\n{'='*60}")
    print(f"  {ckpt_path}")
    print(f"  stage={meta['stage']}  epoch={meta.get('epoch')}  "
          f"val_loss={meta.get('val_loss') or meta.get('best_val_loss')}")
    print(f"{'='*60}")

    # Holdout
    print(f"  Running holdout eval ({len(holdout_rows)} samples)...")
    holdout_results = _run_eval(
        model, holdout_rows, processor,
        emotion_encoder, gender_encoder, age_encoder,
        device, batch_size,
        tasks={"emotion", "gender", "age"},
        desc="  holdout",
    )
    _print_task_table(holdout_results, label="Holdout")

    # KazEmoTTS
    kazemo_results = None
    if kazemo_rows:
        print(f"  Running KazEmoTTS eval ({len(kazemo_rows)} samples)...")
        kazemo_results = _run_eval(
            model, kazemo_rows, processor,
            emotion_encoder, gender_encoder, age_encoder,
            device, batch_size,
            tasks={"emotion"},
            desc="  kazemo",
        )
        _print_task_table(kazemo_results, label="KazEmoTTS")

    result = {
        "meta": meta,
        "encoders": {
            "emotion": sorted(emotion_encoder.keys()),
            "gender":  sorted(gender_encoder.keys()),
            "age":     sorted(age_encoder.keys()),
        },
        "holdout": holdout_results,
        "kazemo":  kazemo_results,
    }

    del model
    del ckpt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def _print_task_table(results: dict, label: str) -> None:
    for task, m in results.items():
        acc = m["accuracy"]
        f1  = m["macro_f1"]
        n   = m["n"]
        print(f"  [{label}] {task:<8}  acc={acc:.4f}  macro_f1={f1:.4f}  n={n}")
        for cls, cls_acc in sorted(m["per_class_accuracy"].items()):
            print(f"           {cls:<14} {cls_acc:.4f}")


# ── summary table ────────────────────────────────────────────────────────────

def _print_summary(all_results: list[dict]) -> None:
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    header = f"  {'Model':<45} {'epoch':>5}  {'holdout_emo':>11}  {'holdout_gen':>11}  {'holdout_age':>11}  {'kazemo_emo':>10}"
    print(header)
    print("  " + "-" * 78)
    for r in all_results:
        name = Path(r["meta"]["path"]).name
        ep   = r["meta"].get("epoch", "?")
        h    = r.get("holdout") or {}
        k    = r.get("kazemo") or {}
        emo  = h.get("emotion", {}).get("accuracy", float("nan"))
        gen  = h.get("gender",  {}).get("accuracy", float("nan"))
        age  = h.get("age",     {}).get("accuracy", float("nan"))
        kemo = k.get("emotion", {}).get("accuracy", float("nan")) if k else float("nan")
        print(f"  {name:<45} {str(ep):>5}  {emo:>11.4f}  {gen:>11.4f}  {age:>11.4f}  {kemo:>10.4f}")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model checkpoints on holdout + KazEmoTTS."
    )
    parser.add_argument("models", nargs="+", help="Paths to .pt checkpoint files")
    parser.add_argument("--output", default="", help="Save JSON results to this path")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kazemo-n", type=int, default=KAZEMO_EVAL_N,
                        help="KazEmoTTS eval samples (default 10 000)")
    parser.add_argument("--kazemo-skip", type=int, default=KAZEMO_TRAIN_N,
                        help="Skip first N KazEmoTTS samples (default 20 000 = training split)")
    parser.add_argument("--no-kazemo", action="store_true",
                        help="Skip KazEmoTTS evaluation")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH),
                        help="Path to validation_subset_manifest.json")
    args = parser.parse_args()

    load_dotenv(_REPO_ROOT / ".env")
    tok = os.environ.get("HF_TOKEN")
    if tok:
        login(token=tok)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load shared data once
    manifest = Path(args.manifest)
    print(f"\nLoading holdout set from {manifest}...")
    holdout_rows = _load_holdout_rows(manifest, HF_BATCH01_ID, HF_BATCH01_CACHE)
    print(f"  {len(holdout_rows)} holdout rows loaded.")

    kazemo_rows: list[dict] | None = None
    if not args.no_kazemo:
        print(f"\nLoading KazEmoTTS eval split "
              f"(skip={args.kazemo_skip}, n={args.kazemo_n})...")
        kazemo_rows = _load_kazemo_rows(
            cache_dir=str(DATA_DIR),
            n=args.kazemo_n,
            skip=args.kazemo_skip,
        )
        print(f"  {len(kazemo_rows)} KazEmoTTS rows loaded.")

    # Evaluate each model
    all_results: list[dict] = []
    for ckpt_str in args.models:
        ckpt_path = Path(ckpt_str)
        if not ckpt_path.exists():
            print(f"\nWARNING: checkpoint not found: {ckpt_path}", file=sys.stderr)
            continue
        try:
            result = evaluate_model(
                ckpt_path, holdout_rows, kazemo_rows, device, args.batch_size
            )
            all_results.append(result)
        except Exception as e:
            import traceback
            print(f"\nERROR evaluating {ckpt_path}: {e}", file=sys.stderr)
            traceback.print_exc()

    if not all_results:
        print("No models evaluated successfully.")
        sys.exit(1)

    _print_summary(all_results)

    # Save
    out_path = Path(args.output) if args.output else (
        _REPO_ROOT / "results" / "model_comparison.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
