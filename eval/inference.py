#!/usr/bin/env python3
"""Inference helpers and batch evaluation for the MultiTaskHubert model."""

import os
import json
from pathlib import Path

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["TORCHCODEC_QUIET"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from datasets import Audio

from loaders.load_data import load, read_audio
from multihead.model import MultiTaskBackbone
from multihead.utils import AudioDataset, MODEL_DIR, SAMPLE_RATE
from utils.config import build_feature_extractor

BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint() -> Path:
    """Return the best available checkpoint path.

    Priority: finetuned model → stage-1 best model.
    """
    finetune = Path(MODEL_DIR) / "finetune" / "best_model_finetuned.pt"
    stage1 = Path(MODEL_DIR) / "best_model.pt"
    if finetune.exists():
        return finetune
    if stage1.exists():
        return stage1
    raise FileNotFoundError(
        "No checkpoint found. Run multihead/train.py (and optionally multihead/finetune.py) first."
    )


def load_model(ckpt_path: Path | None = None, device: torch.device | None = None):
    """Load model and label encoders from a checkpoint.

    Args:
        ckpt_path: Path to .pt checkpoint. Defaults to resolve_checkpoint().
        device:    Target device. Defaults to CUDA if available, else CPU.

    Returns:
        Tuple of (model, emotion_encoder, gender_encoder, age_encoder,
                  id2emotion, id2gender, id2age, processor)
    """
    if ckpt_path is None:
        ckpt_path = resolve_checkpoint()
    if device is None:
        device = DEVICE

    enc_path = Path(MODEL_DIR) / "label_encoders.json"
    if not enc_path.exists():
        raise FileNotFoundError(f"Label encoders not found at {enc_path}")

    with open(enc_path) as f:
        encoders = json.load(f)
    emotion_encoder = encoders["emotion"]
    gender_encoder  = encoders["gender"]
    age_encoder     = encoders["age"]

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

    id2emotion = {v: k for k, v in emotion_encoder.items()}
    id2gender  = {v: k for k, v in gender_encoder.items()}
    id2age     = {v: k for k, v in age_encoder.items()}

    return (model, emotion_encoder, gender_encoder, age_encoder,
            id2emotion, id2gender, id2age, processor)


def run_row_inference(row: dict, model, processor,
                      id2emotion: dict, id2gender: dict, id2age: dict,
                      device: torch.device | None = None,
                      max_length: int = 160000) -> dict:
    """Run inference on a single dataset row.

    Args:
        row:        Dataset row dict (audio must be raw bytes/path, not decoded array).
        model:      Loaded MultiTaskHubert in eval mode.
        processor:  Wav2Vec2FeatureExtractor.
        id2emotion/id2gender/id2age: Index-to-label reverse maps.
        device:     Target device. Defaults to DEVICE.
        max_length: Max audio samples (default 10 s at 16 kHz).

    Returns:
        dict with keys 'emotion', 'gender', 'age', each containing:
            {'pred': str, 'confidence': float, 'top3': [(label, prob), ...]}
    """
    if device is None:
        device = DEVICE

    audio_data, sample_rate = read_audio(row["audio"])

    if sample_rate != SAMPLE_RATE:
        import librosa
        audio_data = librosa.resample(audio_data.astype(np.float32),
                                      orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    if len(audio_data) > max_length:
        audio_data = audio_data[:max_length]
    else:
        audio_data = np.pad(audio_data, (0, max_length - len(audio_data)))

    inputs = processor(audio_data, sampling_rate=SAMPLE_RATE,
                       return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        emotion_logits, gender_logits, age_logits = model(input_values)

    def _top(logits, id2label, k=3):
        probs = F.softmax(logits[0], dim=0)
        topk = probs.topk(min(k, len(probs)))
        return [(id2label[i.item()], round(p.item(), 4))
                for p, i in zip(topk.values, topk.indices)]

    results = {}
    for task, logits, id2label in [
        ("emotion", emotion_logits, id2emotion),
        ("gender",  gender_logits,  id2gender),
        ("age",     age_logits,     id2age),
    ]:
        top = _top(logits, id2label)
        results[task] = {"pred": top[0][0], "confidence": top[0][1], "top3": top}
    return results


def main():
    ckpt_path = resolve_checkpoint()
    print(f"Loading checkpoint: {ckpt_path}")

    (model, emotion_encoder, gender_encoder, age_encoder,
     id2emotion, id2gender, id2age, processor) = load_model(ckpt_path, DEVICE)

    enc_path = Path(MODEL_DIR) / "label_encoders.json"
    dataset = load()
    split = dataset.get("test", dataset.get("validation", dataset.get("val")))
    if split is None:
        split = dataset[list(dataset.keys())[0]]
    if "audio" in split.column_names:
        split = split.cast_column("audio", Audio(decode=False))

    test_ds = AudioDataset(split, processor, emotion_encoder, gender_encoder, age_encoder)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    emotion_correct = gender_correct = age_correct = total = 0
    with torch.no_grad():
        for batch in loader:
            emotion_logits, gender_logits, age_logits = model(batch["input_values"].to(DEVICE))
            emotion_correct += (emotion_logits.argmax(1) == batch["emotion"].to(DEVICE)).sum().item()
            gender_correct  += (gender_logits.argmax(1)  == batch["gender"].to(DEVICE)).sum().item()
            age_correct     += (age_logits.argmax(1)     == batch["age"].to(DEVICE)).sum().item()
            total += batch["emotion"].size(0)

    n = max(total, 1)
    print(
        f"Test n={total} | "
        f"Emotion acc: {emotion_correct / n:.4f} | "
        f"Gender acc: {gender_correct / n:.4f} | "
        f"Age acc: {age_correct / n:.4f}"
    )


if __name__ == "__main__":
    main()
