#!/usr/bin/env python3
"""Gradio demo for MultiTaskHuBERT: emotion, gender, and age from audio.

Usage:
  python eval/demo.py path/to/model1.pt path/to/model2.pt ...
  python eval/demo.py models/finetune/best_model_finetuned.pt --share
"""

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["TORCHCODEC_QUIET"] = "1"

import numpy as np
import gradio as gr
import torch
import torch.nn.functional as F
from multihead.model import MultiTaskBackbone
from multihead.utils import SAMPLE_RATE
from utils.config import build_feature_extractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 160_000  # 10 s at 16 kHz


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _resolve_encoders(ckpt_path: Path) -> Path:
    for candidate in (ckpt_path.parent, ckpt_path.parent.parent):
        p = candidate / "label_encoders.json"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"label_encoders.json not found next to {ckpt_path} or its parent."
    )


def _load_model(ckpt_path: Path):
    """Load one MultiTaskHubert checkpoint.

    Returns (model, id2emotion, id2gender, id2age, label) or raises on error.
    """
    enc_path = _resolve_encoders(ckpt_path)
    with open(enc_path) as f:
        encoders = json.load(f)

    emotion_encoder: dict = encoders["emotion"]
    gender_encoder:  dict = encoders["gender"]
    age_encoder:     dict = encoders["age"]

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
    model = model.to(DEVICE).eval()

    id2emotion = {int(v): k for k, v in emotion_encoder.items()}
    id2gender  = {int(v): k for k, v in gender_encoder.items()}
    id2age     = {int(v): k for k, v in age_encoder.items()}

    label = ckpt_path.name
    return model, id2emotion, id2gender, id2age, label


# ── audio helpers ─────────────────────────────────────────────────────────────

def _prepare_input(audio) -> torch.Tensor | None:
    """Convert Gradio audio input to a (1, MAX_LENGTH) tensor on DEVICE."""
    if audio is None:
        return None

    if isinstance(audio, tuple):
        sr, x = audio
        x = np.array(x, dtype=np.float32)
        if sr is None:
            sr = SAMPLE_RATE
    else:
        import librosa
        x, sr = librosa.load(audio, sr=None, mono=True)

    if x.ndim > 1:
        x = np.mean(x, axis=1)
    x = x.astype(np.float32)

    if sr != SAMPLE_RATE:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SAMPLE_RATE)

    if len(x) > MAX_LENGTH:
        x = x[:MAX_LENGTH]
    else:
        x = np.pad(x, (0, MAX_LENGTH - len(x)))

    processor = build_feature_extractor("facebook/hubert-base-ls960")
    inputs = processor(x, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    return inputs.input_values.to(DEVICE)


def _run_model(model_tuple, input_values: torch.Tensor):
    """Run one model. Returns (emotion_dict, gender_dict, age_dict)."""
    model, id2emotion, id2gender, id2age, _label = model_tuple
    with torch.no_grad():
        emo_logits, gen_logits, age_logits = model(input_values)

    def probs(logits, id2label):
        p = F.softmax(logits[0], dim=0)
        return {id2label[i]: round(p[i].item(), 4) for i in range(len(p))}

    return probs(emo_logits, id2emotion), probs(gen_logits, id2gender), probs(age_logits, id2age)


# ── Gradio app ────────────────────────────────────────────────────────────────

def build_app(model_paths: list[str], share: bool) -> None:
    loaded: list[tuple] = []
    for p in model_paths:
        path = Path(p)
        if not path.exists():
            print(f"WARNING: checkpoint not found, skipping: {path}", file=sys.stderr)
            continue
        try:
            loaded.append(_load_model(path))
            print(f"Loaded: {path}")
        except Exception as e:
            print(f"WARNING: failed to load {path}: {e}", file=sys.stderr)

    if not loaded:
        print("No models loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    n = len(loaded)

    def predict(audio):
        input_values = _prepare_input(audio)
        if input_values is None:
            empty = ({},) * (3 * n)
            return empty
        outputs = []
        for m in loaded:
            e, g, a = _run_model(m, input_values)
            outputs.extend([e, g, a])
        return tuple(outputs)

    with gr.Blocks(title="MultiTaskHuBERT") as demo:
        gr.Markdown("# MultiTaskHuBERT — emotion · gender · age")
        gr.Markdown(
            "Upload or record audio (up to 10 s). "
            f"Running **{n} model{'s' if n > 1 else ''}**."
        )

        audio_input = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Audio")
        submit_btn = gr.Button("Predict")

        output_components = []
        with gr.Row():
            for m in loaded:
                _model, _id2e, _id2g, _id2a, label = m
                num_emo = len(_id2e)
                num_gen = len(_id2g)
                num_age = len(_id2a)
                with gr.Column():
                    gr.Markdown(f"### {label}")
                    output_components.append(gr.Label(num_top_classes=num_emo, label="Emotion"))
                    output_components.append(gr.Label(num_top_classes=num_gen, label="Gender"))
                    output_components.append(gr.Label(num_top_classes=num_age, label="Age"))

        submit_btn.click(fn=predict, inputs=audio_input, outputs=output_components)

    demo.launch(share=share)


def main():
    parser = argparse.ArgumentParser(
        description="Gradio demo for MultiTaskHuBERT. Pass one or more .pt checkpoint paths."
    )
    parser.add_argument(
        "models", nargs="+",
        help="Paths to .pt checkpoint files. label_encoders.json is resolved from the same or parent directory."
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    args = parser.parse_args()

    build_app(args.models, share=args.share)


if __name__ == "__main__":
    main()
