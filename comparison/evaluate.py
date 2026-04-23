#!/usr/bin/env python3
"""Evaluate an emotion2vec variant on the per-corpus test slices of a split dir.

Uses the same split manifests as training (splits/<name>/) so results line up
with what multihead/train.py would see. Inference goes through FunASR's
``AutoModel`` API. Requires ``funasr`` to be installed (not in requirements.txt
by default — install with ``pip install -U funasr``).

Output layout:
    results/comparison/<split_name>/<variant_tail>/
        <corpus>.json          # per-corpus metrics
        summary.json           # aggregate across corpora

Usage:
    python comparison/evaluate.py \
        --split-dir splits/b1_b2_noaug_v1 \
        --variant iic/emotion2vec_plus_large
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

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

# emotion2vec 9-class label set. "other" / "unknown" are catch-all outputs
# that have no ground-truth equivalent — we count them separately and exclude
# them from accuracy so one unusable prediction doesn't get folded in as wrong.
E2V_LABELS = {
    "angry", "disgusted", "fearful", "happy",
    "neutral", "other", "sad", "surprised", "unknown",
}
UNMAPPED_LABELS = {"other", "unknown"}


def _resample(wav: np.ndarray, src_sr: int) -> np.ndarray:
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = wav.astype(np.float32)
    if src_sr != SAMPLE_RATE:
        import librosa
        wav = librosa.resample(wav, orig_sr=src_sr, target_sr=SAMPLE_RATE)
    return wav


def _parse_label(raw: str) -> str:
    # FunASR emotion2vec often returns strings like "0/angry" or "angry/生气".
    # Take the first English-looking token we recognise.
    s = str(raw).strip().lower()
    for piece in s.replace("|", "/").split("/"):
        piece = piece.strip()
        if piece in E2V_LABELS:
            return piece
    # Fallback: try the last token (some builds put the English label last).
    tail = s.split("/")[-1].strip()
    return tail if tail in E2V_LABELS else s


def _predict(model, wav: np.ndarray) -> tuple[str, dict[str, float]]:
    out = model.generate(
        wav,
        granularity="utterance",
        extract_embedding=False,
        disable_pbar=True,
    )
    if isinstance(out, list):
        out = out[0]
    labels_raw = out.get("labels") or []
    scores = out.get("scores") or []
    parsed = [_parse_label(lbl) for lbl in labels_raw]
    score_map = {lbl: float(s) for lbl, s in zip(parsed, scores)}
    # argmax — prefer explicit score pairing, fall back to first label.
    if score_map:
        top = max(score_map.items(), key=lambda kv: kv[1])[0]
    elif parsed:
        top = parsed[0]
    else:
        top = "unknown"
    return top, score_map


def _eval_corpus(
    model,
    corpus_name: str,
    dataset,
    max_samples: int | None,
) -> dict:
    n = len(dataset)
    if max_samples is not None and max_samples > 0:
        n = min(n, max_samples)
        dataset = dataset.select(range(n))

    correct = 0
    counted = 0  # ground-truth rows whose GT label is in the e2v label set
    skipped_gt_oov = 0
    predicted_unmapped = 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in tqdm(dataset, desc=f"{corpus_name}", unit="clip"):
        gt = row.get("emotion")
        if gt is None:
            skipped_gt_oov += 1
            continue
        gt = str(gt).strip().lower()
        if gt not in E2V_LABELS or gt in UNMAPPED_LABELS:
            skipped_gt_oov += 1
            continue

        wav, sr = read_audio(row["audio"])
        wav = _resample(wav, sr)
        pred, _ = _predict(model, wav)

        confusion[gt][pred] += 1
        if pred in UNMAPPED_LABELS:
            predicted_unmapped += 1
            # don't score: model opted out; keep confusion entry for inspection
            continue

        per_class[gt]["total"] += 1
        if pred == gt:
            per_class[gt]["correct"] += 1
            correct += 1
        counted += 1

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split-dir", required=True, help="Path to splits/<name>/ dir")
    ap.add_argument("--variant", required=True, help="FunASR/ModelScope model id, e.g. iic/emotion2vec_plus_large")
    ap.add_argument("--hub", default="hf", choices=["hf", "ms"],
                    help='Model hub: "hf" (HuggingFace) or "ms" (ModelScope). Default hf.')
    ap.add_argument("--out", default=None, help="Output dir. Default: results/comparison/<split>/<variant>/")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap per corpus (useful for quick checks).")
    ap.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    args = ap.parse_args()

    split_dir = Path(args.split_dir).resolve()
    if not (split_dir / "config.yaml").exists():
        raise SystemExit(f"Not a split dir (no config.yaml): {split_dir}")

    variant_tail = args.variant.split("/")[-1]
    out_dir = Path(args.out) if args.out else (
        REPO_ROOT / "results" / "comparison" / split_dir.name / variant_tail
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from funasr import AutoModel
    except ImportError as e:
        raise SystemExit(
            "funasr is not installed. Run: pip install -U funasr\n"
            f"(import error: {e})"
        )

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cmp] variant={args.variant} hub={args.hub} device={device}", flush=True)
    print(f"[cmp] split_dir={split_dir}", flush=True)
    print(f"[cmp] out_dir={out_dir}", flush=True)

    model = AutoModel(model=args.variant, hub=args.hub, device=device, disable_update=True)

    per_corpus = materialize_named_test(split_dir)
    if not per_corpus:
        raise SystemExit(f"No test rows found in {split_dir}")

    aggregate = {
        "split_dir": str(split_dir),
        "variant": args.variant,
        "hub": args.hub,
        "per_corpus": {},
    }
    for corpus, ds in per_corpus.items():
        print(f"[cmp] {corpus}: {len(ds)} test rows", flush=True)
        metrics = _eval_corpus(model, corpus, ds, args.max_samples)
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
