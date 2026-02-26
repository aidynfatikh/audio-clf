#!/usr/bin/env python3
"""Run inference with best model on test set."""

import os
import json
from pathlib import Path

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["TORCHCODEC_QUIET"] = "1"

import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor
from datasets import Audio

from load_data import load, read_audio
from train import MultiTaskHubert, AudioDataset, MODEL_DIR, SAMPLE_RATE

BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    model_dir = Path(MODEL_DIR)
    ckpt_path = model_dir / "best_model.pt"
    enc_path = model_dir / "label_encoders.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    if not enc_path.exists():
        raise FileNotFoundError(f"No encoders at {enc_path}")

    with open(enc_path) as f:
        encoders = json.load(f)
    emotion_encoder = encoders["emotion"]
    gender_encoder = encoders["gender"]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    num_emotions = ckpt["num_emotions"]
    num_genders = ckpt["num_genders"]

    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    dataset = load()
    split = dataset.get("test", dataset.get("validation", dataset.get("val")))
    if split is None:
        split = dataset[list(dataset.keys())[0]]
    if "audio" in split.column_names:
        split = split.cast_column("audio", Audio(decode=False))

    test_ds = AudioDataset(split, processor, emotion_encoder, gender_encoder)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultiTaskHubert(num_emotions=num_emotions, num_genders=num_genders, freeze_backbone=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(DEVICE)
    model.eval()

    emotion_correct = gender_correct = total = 0
    age_ae = 0.0

    with torch.no_grad():
        for batch in loader:
            out = model(batch["input_values"].to(DEVICE))
            emotion_logits, gender_logits, age_pred = out
            emotion_pred = emotion_logits.argmax(dim=1)
            gender_pred = gender_logits.argmax(dim=1)
            emotion_correct += (emotion_pred == batch["emotion"].to(DEVICE)).sum().item()
            gender_correct += (gender_pred == batch["gender"].to(DEVICE)).sum().item()
            age_ae += (age_pred.squeeze() - batch["age"].to(DEVICE)).abs().sum().item()
            total += batch["emotion"].size(0)

    n = max(total, 1)
    print(
        f"Test n={total} | "
        f"Emotion acc: {emotion_correct / n:.4f} | "
        f"Gender acc: {gender_correct / n:.4f} | "
        f"Age MAE: {age_ae / n:.4f}"
    )


if __name__ == "__main__":
    main()
