#!/usr/bin/env python3
"""Evaluate the MultiTaskHubert model on el_audios data (test_pairs.json).

Labels: child is mapped to young for evaluation (no separate child class in practice).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Run from repo root so train/inference are importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["TORCHCODEC_QUIET"] = "1"

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from inference import load_model, resolve_checkpoint, DEVICE
from train import MODEL_DIR, SAMPLE_RATE

MAX_LENGTH = 160000  # 10 s at 16 kHz


def load_audio(path: Path):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), sr


def run_inference(audio_path: Path, model, processor, id2emotion, id2gender, id2age, device):
    audio_data, sample_rate = load_audio(audio_path)
    if sample_rate != SAMPLE_RATE:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if len(audio_data) > MAX_LENGTH:
        audio_data = audio_data[:MAX_LENGTH]
    else:
        audio_data = np.pad(audio_data, (0, MAX_LENGTH - len(audio_data)))

    inputs = processor(audio_data, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        emotion_logits, gender_logits, age_logits = model(input_values)

    def argmax_id(logits):
        return logits[0].argmax().item()

    return {
        "emotion": id2emotion[argmax_id(emotion_logits)],
        "gender": id2gender[argmax_id(gender_logits)],
        "age": id2age[argmax_id(age_logits)],
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate model on el_audios test_pairs.json")
    p.add_argument("--el-audios", type=Path, default=REPO_ROOT / "el_audios", help="el_audios dir (WAVs + info/)")
    p.add_argument("--pairs", type=Path, default=None, help="test_pairs.json path (default: <el_audios>/info/test_pairs.json)")
    p.add_argument("--ckpt", type=Path, default=None, help="Checkpoint path (default: auto)")
    args = p.parse_args()

    el_audios = args.el_audios.resolve()
    pairs_path = args.pairs or (el_audios / "info" / "test_pairs.json")
    if not pairs_path.exists():
        raise SystemExit(f"Not found: {pairs_path}")

    with open(pairs_path, encoding="utf-8") as f:
        pairs = json.load(f)
    if not pairs:
        raise SystemExit("No pairs in test_pairs.json")

    # Load label encoders for mapping labels -> indices
    enc_path = MODEL_DIR / "label_encoders.json"
    if not enc_path.exists():
        raise SystemExit(f"Label encoders not found: {enc_path}")
    with open(enc_path) as f:
        encoders = json.load(f)
    emotion_encoder = encoders["emotion"]
    gender_encoder = encoders["gender"]
    age_encoder = encoders["age"]

    # Child -> young for evaluation labels
    def label_age(age: str) -> str:
        return "young" if age == "child" else age

    ckpt_path = args.ckpt or resolve_checkpoint()
    print(f"Loading model: {ckpt_path}")
    model, _, _, _, id2emotion, id2gender, id2age, processor = load_model(ckpt_path, DEVICE)

    results = []
    emotion_correct = gender_correct = age_correct = 0
    for i, row in enumerate(pairs):
        path = el_audios / row["path"]
        if not path.exists():
            print(f"  Skip (missing): {row['path']}", file=sys.stderr)
            continue
        pred = run_inference(path, model, processor, id2emotion, id2gender, id2age, DEVICE)
        gt_gender = row["gender"]
        gt_age = label_age(row["age"])
        gt_emotion = row["emotion"]

        e_ok = pred["emotion"] == gt_emotion
        g_ok = pred["gender"] == gt_gender
        a_ok = pred["age"] == gt_age
        emotion_correct += int(e_ok)
        gender_correct += int(g_ok)
        age_correct += int(a_ok)

        results.append({
            "path": row["path"],
            "gender": {"gt": gt_gender, "pred": pred["gender"], "ok": g_ok},
            "age": {"gt": gt_age, "pred": pred["age"], "ok": a_ok},
            "emotion": {"gt": gt_emotion, "pred": pred["emotion"], "ok": e_ok},
        })

    n = len(results)
    if n == 0:
        raise SystemExit("No files evaluated.")
    print(f"Evaluated {n} samples (child → young for age label)")
    print(f"  Emotion acc: {emotion_correct}/{n} = {emotion_correct/n:.4f}")
    print(f"  Gender  acc: {gender_correct}/{n} = {gender_correct/n:.4f}")
    print(f"  Age     acc: {age_correct}/{n} = {age_correct/n:.4f}")

    out_json = el_audios / "info" / "eval_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "n": n,
            "accuracy": {"emotion": emotion_correct / n, "gender": gender_correct / n, "age": age_correct / n},
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_json}")


if __name__ == "__main__":
    main()
