#!/usr/bin/env python3
"""Gradio demo for MultiTaskHuBERT and Single-Head HuBERT: emotion, gender, and age from audio.

Runs inference with 4 models: multi-head best, multi-head latest, single-head (emotion) best,
single-head (emotion) latest.
"""

import os
import json
import sys
import argparse
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["TORCHCODEC_QUIET"] = "1"

import numpy as np
import gradio as gr
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor

from train import MultiTaskHubert, MODEL_DIR, SAMPLE_RATE

# Single-head (emotion) paths
SINGLE_HEAD_FEATURE = "emotion"
SINGLE_HEAD_MODEL_DIR = _REPO_ROOT / "single_head" / "models" / SINGLE_HEAD_FEATURE
SINGLE_HEAD_FINETUNE_DIR = SINGLE_HEAD_MODEL_DIR / "finetune"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 160000  # 10 s at 16 kHz

_models_data = None


def _load_multi_model(ckpt_path: Path, encoders: dict):
    """Load one MultiTaskHubert checkpoint. Returns (model, id2emotion, id2gender, id2age)."""
    emotion_encoder = encoders["emotion"]
    gender_encoder = encoders["gender"]
    age_encoder = encoders["age"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MultiTaskHubert(
        num_emotions=ckpt["num_emotions"],
        num_genders=ckpt["num_genders"],
        num_ages=ckpt["num_ages"],
        freeze_backbone=True,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(DEVICE).eval()
    id2emotion = {int(v): k for k, v in emotion_encoder.items()}
    id2gender = {int(v): k for k, v in gender_encoder.items()}
    id2age = {int(v): k for k, v in age_encoder.items()}
    return model, id2emotion, id2gender, id2age


def _load_single_head_model(ckpt_path: Path, encoders: dict):
    """Load one SingleHeadHubert (emotion) checkpoint. Returns (model, id2emotion)."""
    from single_head.train_single import SingleHeadHubert

    emotion_encoder = encoders[SINGLE_HEAD_FEATURE]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SingleHeadHubert(
        feature=ckpt["feature"],
        num_classes=ckpt["num_classes"],
        freeze_backbone=True,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(DEVICE).eval()
    id2emotion = {int(v): k for k, v in emotion_encoder.items()}
    return model, id2emotion


def _load_models():
    global _models_data
    if _models_data is not None:
        return _models_data

    # Multi-head encoders and checkpoints (optional; skip missing files)
    multi_encoders = None
    enc_path = MODEL_DIR / "label_encoders.json"
    if enc_path.exists():
        with open(enc_path) as f:
            multi_encoders = json.load(f)

    multi_best_path = MODEL_DIR / "finetune" / "best_model_finetuned.pt"
    multi_latest_path = MODEL_DIR / "finetune" / "latest_checkpoint_finetune.pt"

    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

    multi_best_data = (
        _load_multi_model(multi_best_path, multi_encoders)
        if multi_encoders is not None and multi_best_path.exists()
        else None
    )
    multi_latest_data = (
        _load_multi_model(multi_latest_path, multi_encoders)
        if multi_encoders is not None and multi_latest_path.exists()
        else None
    )

    # Single-head (emotion) encoders and checkpoints (optional; skip missing files)
    single_encoders = None
    single_enc_path = SINGLE_HEAD_MODEL_DIR / "label_encoders.json"
    if single_enc_path.exists():
        with open(single_enc_path) as f:
            single_encoders = json.load(f)

    single_best_path = SINGLE_HEAD_FINETUNE_DIR / "best_model_finetuned.pt"
    single_latest_path = SINGLE_HEAD_FINETUNE_DIR / "latest_checkpoint_finetune.pt"

    single_best_data = (
        _load_single_head_model(single_best_path, single_encoders)
        if single_encoders is not None and single_best_path.exists()
        else None
    )
    single_latest_data = (
        _load_single_head_model(single_latest_path, single_encoders)
        if single_encoders is not None and single_latest_path.exists()
        else None
    )

    _models_data = (
        processor,
        multi_best_data,
        multi_latest_data,
        single_best_data,
        single_latest_data,
    )
    return _models_data


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


def _run_multi_model(model_data, input_values):
    """Run multi-head model. Returns (emotion_dict, gender_dict, age_dict)."""
    if model_data is None:
        return {}, {}, {}
    model, id2emotion, id2gender, id2age = model_data
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


def _run_single_head_model(model_data, input_values):
    """Run single-head (emotion) model. Returns emotion_dict only."""
    if model_data is None:
        return {}
    model, id2emotion = model_data
    with torch.no_grad():
        logits = model(input_values)
    probs = F.softmax(logits[0], dim=0)
    return {id2emotion[i]: round(p.item(), 4) for i, p in enumerate(probs)}


def predict_all(audio):
    """Run all 4 models on one audio input.

    Returns 8 gr.Label-compatible dicts:
      multi_best (e, g, a), multi_latest (e, g, a), single_best (e), single_latest (e).
    """
    empty = ({}, {}, {}, {}, {}, {}, {}, {})
    if audio is None:
        return empty
    (
        processor,
        multi_best_data,
        multi_latest_data,
        single_best_data,
        single_latest_data,
    ) = _load_models()
    sr, waveform = _process_audio(audio)
    if waveform is None or len(waveform) == 0:
        return empty

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

    multi_best_e, multi_best_g, multi_best_a = _run_multi_model(multi_best_data, input_values)
    multi_latest_e, multi_latest_g, multi_latest_a = _run_multi_model(multi_latest_data, input_values)
    single_best_e = _run_single_head_model(single_best_data, input_values)
    single_latest_e = _run_single_head_model(single_latest_data, input_values)

    return (
        multi_best_e,
        multi_best_g,
        multi_best_a,
        multi_latest_e,
        multi_latest_g,
        multi_latest_a,
        single_best_e,
        single_latest_e,
    )


def predict_best(audio):
    """Run only the multi-head best model on one audio input."""
    (
        multi_best_e,
        multi_best_g,
        multi_best_a,
        _multi_latest_e,
        _multi_latest_g,
        _multi_latest_a,
        _single_best_e,
        _single_latest_e,
    ) = predict_all(audio)
    return multi_best_e, multi_best_g, multi_best_a


def main():
    parser = argparse.ArgumentParser(
        description="Gradio demo for MultiTaskHuBERT and Single-Head HuBERT."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Show all models (multi-head best/latest and single-head best/latest). "
            "By default only the multi-head best model is run and displayed."
        ),
    )
    args = parser.parse_args()

    _load_models()  # Populate cache; missing models are skipped gracefully

    title = "HuBERT — Multi-Head & Single-Head"
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        if args.all:
            gr.Markdown(
                "Upload or record audio (up to 10 s). **4 models** run inference: "
                "multi-head best/latest (emotion, gender, age) and single-head (emotion) best/latest."
            )
        else:
            gr.Markdown(
                "Upload or record audio (up to 10 s). "
                "By default this demo runs only the **multi-head best** model "
                "(emotion, gender, age). Use `--all` to also run latest and single-head models."
            )

        audio_input = gr.Audio(
            sources=["upload", "microphone"], type="numpy", label="Audio Input"
        )
        submit_btn = gr.Button("Predict")

        with gr.Row():
            # Column 1: Multi-head best (always shown)
            with gr.Column():
                gr.Markdown("### Multi-head best")
                gr.Markdown("`models/finetune/best_model_finetuned.pt`")
                multi_best_emotion = gr.Label(num_top_classes=7, label="Emotion")
                multi_best_gender = gr.Label(num_top_classes=4, label="Gender")
                multi_best_age = gr.Label(num_top_classes=4, label="Age")

            if args.all:
                # Column 2: Multi-head latest
                with gr.Column():
                    gr.Markdown("### Multi-head latest")
                    gr.Markdown("`models/finetune/latest_checkpoint_finetune.pt`")
                    multi_latest_emotion = gr.Label(
                        num_top_classes=7, label="Emotion"
                    )
                    multi_latest_gender = gr.Label(
                        num_top_classes=4, label="Gender"
                    )
                    multi_latest_age = gr.Label(num_top_classes=4, label="Age")
                # Column 3: Single-head best (emotion)
                with gr.Column():
                    gr.Markdown("### Single-head best (emotion)")
                    gr.Markdown(
                        "`single_head/models/emotion/finetune/best_model_finetuned.pt`"
                    )
                    single_best_emotion = gr.Label(
                        num_top_classes=7, label="Emotion"
                    )
                # Column 4: Single-head latest (emotion)
                with gr.Column():
                    gr.Markdown("### Single-head latest (emotion)")
                    gr.Markdown(
                        "`single_head/models/emotion/finetune/latest_checkpoint_finetune.pt`"
                    )
                    single_latest_emotion = gr.Label(
                        num_top_classes=7, label="Emotion"
                    )

        if args.all:
            submit_btn.click(
                fn=predict_all,
                inputs=audio_input,
                outputs=[
                    multi_best_emotion,
                    multi_best_gender,
                    multi_best_age,
                    multi_latest_emotion,
                    multi_latest_gender,
                    multi_latest_age,
                    single_best_emotion,
                    single_latest_emotion,
                ],
            )
        else:
            submit_btn.click(
                fn=predict_best,
                inputs=audio_input,
                outputs=[
                    multi_best_emotion,
                    multi_best_gender,
                    multi_best_age,
                ],
            )

    demo.launch(share=True)


if __name__ == "__main__":
    main()
