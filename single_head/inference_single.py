#!/usr/bin/env python3
"""Inference for single-head HuBERT (one feature per model)."""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo root is on path when running from single_head/
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["TORCHCODEC_QUIET"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor
from datasets import Audio

from load_data import load, read_audio
from train_single import SingleHeadHubert, AudioDataset, build_label_encoders, FEATURES

BATCH_SIZE = 8
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base dir for single-head models: single_head/models/{feature}
MODEL_BASE = Path(__file__).resolve().parent / "models"


def resolve_checkpoint(feature: str) -> Path:
    """Best available checkpoint for the given feature: finetuned → stage-1."""
    finetune = MODEL_BASE / feature / "finetune" / "best_model_finetuned.pt"
    stage1 = MODEL_BASE / feature / "best_model.pt"
    if finetune.exists():
        return finetune
    if stage1.exists():
        return stage1
    raise FileNotFoundError(
        f"No checkpoint found for feature '{feature}'. "
        f"Run: python single_head/train_single.py --feature {feature}"
    )


def load_model(feature: str, ckpt_path: Path | None = None, device: torch.device | None = None):
    """Load single-head model and label encoder for the given feature."""
    if ckpt_path is None:
        ckpt_path = resolve_checkpoint(feature)
    if device is None:
        device = DEVICE

    MODEL_DIR = MODEL_BASE / feature
    enc_path = MODEL_DIR / "label_encoders.json"
    if not enc_path.exists():
        raise FileNotFoundError(f"Label encoders not found at {enc_path}")

    with open(enc_path) as f:
        encoders = json.load(f)
    encoder = encoders[feature]
    id2label = {v: k for k, v in encoder.items()}

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SingleHeadHubert(
        feature=ckpt["feature"],
        num_classes=ckpt["num_classes"],
        freeze_backbone=True,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device).eval()

    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    return model, encoder, id2label, processor


def run_row_inference(row: dict, model, processor, id2label: dict,
                      device: torch.device | None = None,
                      max_length: int = 160000) -> dict:
    """Run inference on a single row. Returns dict with 'pred', 'confidence', 'top3' for the feature."""
    if device is None:
        device = DEVICE

    audio_data, sample_rate = read_audio(row["audio"])
    if sample_rate != SAMPLE_RATE:
        import librosa
        audio_data = librosa.resample(
            audio_data.astype(np.float32),
            orig_sr=sample_rate, target_sr=SAMPLE_RATE
        )
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    if len(audio_data) > max_length:
        audio_data = audio_data[:max_length]
    else:
        audio_data = np.pad(audio_data, (0, max_length - len(audio_data)))

    inputs = processor(audio_data, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values)

    probs = F.softmax(logits[0], dim=0)
    k = min(3, len(probs))
    topk = probs.topk(k)
    top3 = [(id2label[i.item()], round(p.item(), 4)) for p, i in zip(topk.values, topk.indices)]
    return {"pred": top3[0][0], "confidence": top3[0][1], "top3": top3}


def main():
    parser = argparse.ArgumentParser(description="Run single-head HuBERT inference.")
    parser.add_argument("--feature", default="emotion", choices=list(FEATURES), help="Feature to evaluate.")
    args = parser.parse_args()
    feature = args.feature

    ckpt_path = resolve_checkpoint(feature)
    print(f"Loading checkpoint: {ckpt_path}")

    model, encoder, id2label, processor = load_model(feature, ckpt_path, DEVICE)
    # Need all encoders for AudioDataset
    dataset = load()
    emotion_encoder, gender_encoder, age_encoder = build_label_encoders(dataset)

    split = dataset.get("test", dataset.get("validation", dataset.get("val")))
    if split is None:
        split = dataset[list(dataset.keys())[0]]
    if "audio" in split.column_names:
        split = split.cast_column("audio", Audio(decode=False))

    test_ds = AudioDataset(split, processor, emotion_encoder, gender_encoder, age_encoder)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_values"].to(DEVICE))
            labels = batch[feature].to(DEVICE)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    n = max(total, 1)
    print(f"Test n={total} | {feature} acc: {correct / n:.4f}")


if __name__ == "__main__":
    main()
