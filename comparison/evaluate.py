#!/usr/bin/env python3
"""Evaluate a speech-emotion model on per-corpus test data.

Source (exactly one):
  --split-dir <dir>    splits/<name>/ — per-corpus test slices from the builder.
  --hf-dataset <id>    HuggingFace dataset id with a ``data.jsonl`` manifest
                       and ``audio`` + ``emotion`` fields (single corpus). All
                       referenced .wav files are prefetched into the HF hub
                       cache up front; iteration then reads purely from disk.

Model (exactly one):
  --variant <id>       FunASR emotion2vec model id (e.g. iic/emotion2vec_plus_large).
                       Requires ``pip install -U funasr``.
  --checkpoint <path>  Local MultiTaskBackbone checkpoint (.pt). Label encoders
                       are loaded from a ``label_encoders.json`` next to the
                       checkpoint (falling back to its parent dir).

Output layout:
    results/comparison/<source_tag>/<model_tag>/
        <corpus>.json          # per-corpus metrics
        summary.json           # aggregate across corpora

Usage:
    python comparison/evaluate.py --split-dir splits/b1_b2_noaug_v1 \\
        --variant iic/emotion2vec_plus_large
    python comparison/evaluate.py --hf-dataset umutkkgz/tr-full-dataset \\
        --max-samples 5000 --checkpoint results/.../best_model_finetuned.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

os.environ.setdefault("DATASETS_AUDIO_BACKEND", "soundfile")
os.environ.setdefault("TORCHCODEC_QUIET", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from tqdm import tqdm

from loaders.load_data import read_audio
from splits.materialize import materialize_named_test

SAMPLE_RATE = 16000
MAX_AUDIO_SAMPLES = 160000  # 10s @ 16kHz, matches eval/kazemo_evaluate.py

# emotion2vec 9-class label set. "other"/"unknown" have no GT equivalent — we
# count them separately and exclude them from accuracy so an "opt-out" from the
# model isn't folded in as wrong.
E2V_LABELS = {
    "angry", "disgusted", "fearful", "happy",
    "neutral", "other", "sad", "surprised", "unknown",
}
UNMAPPED_PRED = {"other", "unknown"}

PredictBatch = Callable[[list[np.ndarray]], list[str]]


# ── Audio helpers ───────────────────────────────────────────────────────────

def _resample(wav: np.ndarray, src_sr: int) -> np.ndarray:
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = wav.astype(np.float32)
    if src_sr != SAMPLE_RATE:
        import librosa
        wav = librosa.resample(wav, orig_sr=src_sr, target_sr=SAMPLE_RATE)
    return wav


def _load_row_waveform(row) -> np.ndarray:
    wav, sr = read_audio(row["audio"])
    return _resample(wav, sr)


def _fit_length(wav: np.ndarray) -> np.ndarray:
    if len(wav) > MAX_AUDIO_SAMPLES:
        return wav[:MAX_AUDIO_SAMPLES]
    return np.pad(wav, (0, MAX_AUDIO_SAMPLES - len(wav)))


# ── emotion2vec (FunASR) predictor ──────────────────────────────────────────

def _parse_e2v_label(raw: str) -> str:
    s = str(raw).strip().lower()
    for piece in s.replace("|", "/").split("/"):
        piece = piece.strip()
        if piece in E2V_LABELS:
            return piece
    tail = s.split("/")[-1].strip()
    return tail if tail in E2V_LABELS else s


def _build_e2v_predictor(variant: str, hub: str, device: str) -> PredictBatch:
    try:
        from funasr import AutoModel
    except ImportError as e:
        raise SystemExit(
            "funasr is not installed. Run: pip install -U funasr\n"
            f"(import error: {e})"
        )
    model = AutoModel(model=variant, hub=hub, device=device, disable_update=True)

    def predict_one(wav: np.ndarray) -> str:
        out = model.generate(
            wav, granularity="utterance", extract_embedding=False, disable_pbar=True,
        )
        if isinstance(out, list):
            out = out[0]
        labels_raw = out.get("labels") or []
        scores = out.get("scores") or []
        parsed = [_parse_e2v_label(lbl) for lbl in labels_raw]
        if scores and parsed:
            return max(zip(parsed, scores), key=lambda kv: float(kv[1]))[0]
        return parsed[0] if parsed else "unknown"

    # FunASR's generate() doesn't batch cleanly across its model zoo; loop.
    def predict_batch(wavs: list[np.ndarray]) -> list[str]:
        return [predict_one(w) for w in wavs]

    return predict_batch


# ── Our model (MultiTaskBackbone) predictor ─────────────────────────────────

def _find_label_encoders(ckpt_path: Path) -> Path:
    for cand in (ckpt_path.parent / "label_encoders.json",
                 ckpt_path.parent.parent / "label_encoders.json"):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"label_encoders.json not found next to {ckpt_path} or its parent."
    )


def _build_our_predictor(ckpt_path: Path, device: str) -> tuple[PredictBatch, dict]:
    import torch
    from multihead.model import MultiTaskBackbone
    from utils.config import build_feature_extractor

    enc_path = _find_label_encoders(ckpt_path)
    with open(enc_path) as f:
        encoders = json.load(f)
    emotion_encoder: dict[str, int] = encoders["emotion"]
    id2emotion = {v: k for k, v in emotion_encoder.items()}

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone_name = ckpt.get("backbone_name", "hubert_base")
    pretrained = ckpt.get("pretrained", "facebook/hubert-base-ls960")
    model = MultiTaskBackbone(
        num_emotions=ckpt["num_emotions"],
        num_genders=ckpt["num_genders"],
        num_ages=ckpt["num_ages"],
        freeze_backbone=True,
        backbone_name=backbone_name,
        pretrained=pretrained,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device).eval()
    processor = build_feature_extractor(pretrained)
    torch_device = torch.device(device)

    def predict_batch(wavs: list[np.ndarray]) -> list[str]:
        fitted = [_fit_length(w) for w in wavs]
        inputs = processor(
            fitted, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True,
        )
        iv = inputs.input_values.to(torch_device)
        with torch.no_grad():
            emo_logits, _, _ = model(iv)
        idxs = emo_logits.argmax(dim=1).tolist()
        return [id2emotion[int(i)] for i in idxs]

    meta = {
        "checkpoint": str(ckpt_path),
        "label_encoders": str(enc_path),
        "backbone_name": backbone_name,
        "pretrained": pretrained,
        "emotion_classes": sorted(emotion_encoder.keys()),
    }
    return predict_batch, meta


# ── Per-corpus scoring loop (predictor-agnostic, batched) ───────────────────

def _eval_corpus(
    predict_batch: PredictBatch,
    corpus_name: str,
    dataset,
    max_samples: int | None,
    batch_size: int,
) -> dict:
    n = len(dataset)
    if max_samples is not None and max_samples > 0:
        n = min(n, max_samples)
        dataset = dataset.select(range(n))

    correct = 0
    counted = 0
    skipped_gt_oov = 0
    predicted_unmapped = 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    pbar = tqdm(total=n, desc=corpus_name, unit="clip")
    batch_wavs: list[np.ndarray] = []
    batch_gts: list[str] = []

    def flush():
        nonlocal correct, counted, predicted_unmapped
        if not batch_wavs:
            return
        preds = predict_batch(batch_wavs)
        for gt, pred in zip(batch_gts, preds):
            confusion[gt][pred] += 1
            if pred in UNMAPPED_PRED:
                predicted_unmapped += 1
                continue
            per_class[gt]["total"] += 1
            if pred == gt:
                per_class[gt]["correct"] += 1
                correct += 1
            counted += 1
        pbar.update(len(batch_wavs))
        batch_wavs.clear()
        batch_gts.clear()

    for row in dataset:
        gt = row.get("emotion")
        if gt is None:
            skipped_gt_oov += 1
            pbar.update(1)
            continue
        gt = str(gt).strip().lower()
        if gt in UNMAPPED_PRED:
            skipped_gt_oov += 1
            pbar.update(1)
            continue

        batch_wavs.append(_load_row_waveform(row))
        batch_gts.append(gt)
        if len(batch_wavs) >= batch_size:
            flush()
    flush()
    pbar.close()

    acc = correct / counted if counted else 0.0
    per_class_acc = {
        k: (v["correct"] / v["total"] if v["total"] else 0.0)
        for k, v in sorted(per_class.items())
    }
    support = {k: int(v["total"]) for k, v in sorted(per_class.items())}
    confusion_out = {
        gt: dict(sorted(preds.items())) for gt, preds in sorted(confusion.items())
    }

    return {
        "corpus": corpus_name,
        "n_total": int(n),
        "n_scored": counted,
        "n_skipped_gt_oov": skipped_gt_oov,
        "n_predicted_unmapped": predicted_unmapped,
        "accuracy": acc,
        "per_class_accuracy": per_class_acc,
        "support": support,
        "confusion": confusion_out,
    }


# ── Source loaders ──────────────────────────────────────────────────────────

def _checkpoint_tag(ckpt_path: Path) -> str:
    # e.g. models/exp1-.../finetune/best_model_finetuned.pt
    # → exp1-...__finetune__best_model_finetuned
    parts = tuple(p for p in ckpt_path.with_suffix("").parts if p and p != "/")
    if "models" in parts:
        parts = parts[parts.index("models") + 1:]
    elif "results" in parts:
        parts = parts[parts.index("results") + 1:]
    else:
        parts = parts[-3:]
    return "__".join(parts) or ckpt_path.stem


def _load_hf_dataset_as_corpus(
    hf_id: str,
    max_samples: int | None,
    manifest: str,
    prefetch_workers: int,
):
    """Return a Dataset with ``audio`` + ``emotion`` columns.

    Auto-detects the loading strategy:
      * **Manifest-backed** (``data.jsonl`` at repo root, per-file audio refs):
        prefetch every referenced .wav into the HF cache, then build a Dataset
        from those local paths. Used by e.g. ``umutkkgz/tr-full-dataset``.
      * **Parquet/script-backed** (audio bytes embedded in shards):
        ``load_dataset`` downloads the shards up front and iteration hits local
        disk. Used by e.g. ``Martingkc/processed_audio_tr-with-emotions``.
    """
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(repo_id=hf_id, repo_type="dataset", filename=manifest)
        has_manifest = True
    except Exception:
        has_manifest = False

    if has_manifest:
        from datasets import Dataset
        from comparison.prefetch_hf import prefetch_hf_audio
        rows = prefetch_hf_audio(
            hf_id, max_samples, manifest=manifest, max_workers=prefetch_workers,
        )
        if not rows:
            raise SystemExit(f"No usable rows from {hf_id} manifest {manifest!r}")
        if "emotion" not in rows[0]:
            raise SystemExit(
                f"{hf_id} manifest has no 'emotion' field (keys: {sorted(rows[0])})"
            )
        return Dataset.from_list(rows)

    # Parquet/script flow — audio is inside shards, no prefetch needed.
    from datasets import Audio, load_dataset
    split_spec = "train"
    if max_samples is not None and max_samples > 0:
        split_spec = f"train[:{max_samples}]"
    print(f"[cmp] no {manifest!r} in {hf_id} — using load_dataset({split_spec!r})", flush=True)
    ds = load_dataset(hf_id, split=split_spec)
    if "emotion" not in ds.column_names:
        raise SystemExit(f"{hf_id} has no 'emotion' column (columns: {ds.column_names})")
    if "audio" in ds.column_names:
        # decode=False → rows carry {'bytes': ..., 'path': ...}, which our
        # read_audio() handles identically to the split-dir and manifest flows.
        ds = ds.cast_column("audio", Audio(decode=False))
    return ds


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--split-dir", help="Path to splits/<name>/ dir (per-corpus test slices)")
    src.add_argument("--hf-dataset", help="HF dataset id with a data.jsonl manifest, e.g. umutkkgz/tr-full-dataset")
    ap.add_argument("--corpus-name", default=None,
                    help="Corpus name used in output filenames. Default: last segment of --hf-dataset.")
    model_group = ap.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--variant", help="FunASR model id, e.g. iic/emotion2vec_plus_large")
    model_group.add_argument("--checkpoint", help="Path to our .pt checkpoint")
    ap.add_argument("--hub", default="hf", choices=["hf", "ms"],
                    help='Hub for --variant: "hf" or "ms". Default hf.')
    ap.add_argument("--out", default=None, help="Output dir. Default: results/comparison/<source>/<tag>/")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap per corpus (useful for quick checks). For --hf-dataset also caps prefetch.")
    ap.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Inference batch size (default 16). FunASR --variant runs effectively batch=1.")
    ap.add_argument("--hf-manifest", default="data.jsonl",
                    help='Manifest filename inside --hf-dataset (default "data.jsonl")')
    ap.add_argument("--prefetch-workers", type=int, default=16,
                    help="Parallel download workers for --hf-dataset audio prefetch (default 16)")
    args = ap.parse_args()

    if args.split_dir:
        split_dir = Path(args.split_dir).resolve()
        if not (split_dir / "config.yaml").exists():
            raise SystemExit(f"Not a split dir (no config.yaml): {split_dir}")
        source_tag = split_dir.name
        per_corpus = materialize_named_test(split_dir)
        if not per_corpus:
            raise SystemExit(f"No test rows found in {split_dir}")
        source_meta = {"kind": "split_dir", "split_dir": str(split_dir)}
    else:
        source_tag = args.hf_dataset.replace("/", "__")
        corpus_name = args.corpus_name or args.hf_dataset.split("/")[-1]
        ds = _load_hf_dataset_as_corpus(
            args.hf_dataset, args.max_samples,
            manifest=args.hf_manifest, prefetch_workers=args.prefetch_workers,
        )
        per_corpus = {corpus_name: ds}
        source_meta = {
            "kind": "hf_dataset",
            "hf_dataset": args.hf_dataset,
            "manifest": args.hf_manifest,
        }

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.variant:
        tag = args.variant.split("/")[-1]
        predict_batch = _build_e2v_predictor(args.variant, args.hub, device)
        model_meta = {"kind": "emotion2vec", "variant": args.variant, "hub": args.hub}
    else:
        ckpt_path = Path(args.checkpoint).resolve()
        if not ckpt_path.exists():
            raise SystemExit(f"Checkpoint not found: {ckpt_path}")
        tag = _checkpoint_tag(ckpt_path)
        predict_batch, meta = _build_our_predictor(ckpt_path, device)
        model_meta = {"kind": "multitask_backbone", **meta}

    out_dir = Path(args.out) if args.out else (
        REPO_ROOT / "results" / "comparison" / source_tag / tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cmp] model={model_meta}", flush=True)
    print(f"[cmp] source={source_meta} device={device} batch_size={args.batch_size}", flush=True)
    print(f"[cmp] out_dir={out_dir}", flush=True)

    aggregate = {
        "source": source_meta,
        "model": model_meta,
        "per_corpus": {},
    }
    for corpus, ds in per_corpus.items():
        print(f"[cmp] {corpus}: {len(ds)} test rows", flush=True)
        metrics = _eval_corpus(predict_batch, corpus, ds, args.max_samples, args.batch_size)
        (out_dir / f"{corpus}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        aggregate["per_corpus"][corpus] = {
            "n_scored": metrics["n_scored"],
            "accuracy": metrics["accuracy"],
            "n_predicted_unmapped": metrics["n_predicted_unmapped"],
            "n_skipped_gt_oov": metrics["n_skipped_gt_oov"],
        }
        print(
            f"[cmp] {corpus}: acc={metrics['accuracy']:.4f} "
            f"scored={metrics['n_scored']} "
            f"oov_gt={metrics['n_skipped_gt_oov']} "
            f"unmapped_pred={metrics['n_predicted_unmapped']}",
            flush=True,
        )

    (out_dir / "summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"[cmp] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
