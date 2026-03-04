#!/usr/bin/env python3
"""Gradio demo for MultiTaskHuBERT: emotion, gender, and age from audio."""

import os
import json
from pathlib import Path

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["TORCHCODEC_QUIET"] = "1"

import numpy as np
import gradio as gr
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor

from train import MultiTaskHubert, MODEL_DIR, SAMPLE_RATE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 160000  # 10 s at 16 kHz

_model_data = None


def _load_model():
    global _model_data
    if _model_data is not None:
        return _model_data
    enc_path = MODEL_DIR / "label_encoders.json"
    if not enc_path.exists():
        raise FileNotFoundError(
            "Label encoders not found. Run train.py first and save encoders to models/label_encoders.json"
        )
    with open(enc_path) as f:
        encoders = json.load(f)
    emotion_encoder = encoders["emotion"]
    gender_encoder = encoders["gender"]
    age_encoder = encoders["age"]

    # Prefer finetuned checkpoint
    finetune = MODEL_DIR / "finetune" / "best_model_finetuned.pt"
    stage1 = MODEL_DIR / "best_model.pt"
    ckpt_path = finetune if finetune.exists() else stage1
    if not ckpt_path.exists():
        raise FileNotFoundError("No checkpoint found. Run train.py (and optionally finetune.py) first.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MultiTaskHubert(
        num_emotions=ckpt["num_emotions"],
        num_genders=ckpt["num_genders"],
        num_ages=ckpt["num_ages"],
        freeze_backbone=True,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(DEVICE).eval()

    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    id2emotion = {int(v): k for k, v in emotion_encoder.items()}
    id2gender = {int(v): k for k, v in gender_encoder.items()}
    id2age = {int(v): k for k, v in age_encoder.items()}

    _model_data = (model, processor, id2emotion, id2gender, id2age)
    return _model_data


def _process_audio(audio):
    """Convert Gradio audio input to (sr, mono float32 array)."""
    if audio is None:
        return None, None
    if isinstance(audio, tuple):
        sr, x = audio
        x = np.array(x, dtype=np.float32)
        if sr is None:
            sr = SAMPLE_RATE
    else:
        import librosa
        x, sr = librosa.load(audio, sr=None, mono=True)
    if len(x.shape) > 1:
        x = np.mean(x, axis=1)
    return sr, x


def predict(audio):
    """Run model on one audio input. Returns (emotion_probs, gender_probs, age_probs) for gr.Label."""
    if audio is None:
        return {}, {}, {}
    model, processor, id2emotion, id2gender, id2age = _load_model()
    sr, waveform = _process_audio(audio)
    if waveform is None or len(waveform) == 0:
        return {}, {}, {}

    if sr != SAMPLE_RATE:
        import librosa
        waveform = librosa.resample(
            waveform.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE
        )
    if len(waveform) > MAX_LENGTH:
        waveform = waveform[:MAX_LENGTH]
    else:
        waveform = np.pad(waveform, (0, MAX_LENGTH - len(waveform)))

    inputs = processor(
        waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
    )
    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        emotion_logits, gender_logits, age_logits = model(input_values)

    def to_label_dict(logits, id2label):
        probs = F.softmax(logits[0], dim=0)
        return {id2label[i]: round(p.item(), 4) for i, p in enumerate(probs)}

    return (
        to_label_dict(emotion_logits, id2emotion),
        to_label_dict(gender_logits, id2gender),
        to_label_dict(age_logits, id2age),
    )


def main():
    _load_model()  # Fail fast if checkpoint/encoders missing
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Audio(sources=["upload", "microphone"], type="numpy", label="Audio"),
        outputs=[
            gr.Label(num_top_classes=7, label="Emotion"),
            gr.Label(num_top_classes=4, label="Gender"),
            gr.Label(num_top_classes=4, label="Age"),
        ],
        title="Multi-Task HuBERT — Emotion, Gender & Age",
        description="Upload or record audio (up to 10 s). Model predicts emotion, gender, and age category.",
    )
    demo.launch(share=True)


if __name__ == "__main__":
    main()
