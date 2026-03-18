#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["TORCHCODEC_QUIET"] = "1"

# Allow running via `python kazemo/evaluate.py` (not only `python -m kazemo.evaluate`).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from inference import load_model, resolve_checkpoint
from load_data import read_audio
from train import MODEL_DIR, SAMPLE_RATE

from kazemo.load_data import load_kazemotts


class KazEmoEmotionDataset(Dataset):
    def __init__(self, dataset_split, processor, emotion_encoder: dict, max_length: int = 160000):
        self.data = dataset_split
        self.processor = processor
        self.emotion_encoder = emotion_encoder
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]

        audio_data, sample_rate = read_audio(row["audio"])
        if sample_rate != SAMPLE_RATE:
            import librosa

            audio_data = librosa.resample(
                audio_data.astype(np.float32), orig_sr=sample_rate, target_sr=SAMPLE_RATE
            )
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        if len(audio_data) > self.max_length:
            audio_data = audio_data[: self.max_length]
        else:
            audio_data = np.pad(audio_data, (0, self.max_length - len(audio_data)))

        inputs = self.processor(
            audio_data, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.squeeze(0)

        emo = row.get("emotion")
        if emo not in self.emotion_encoder:
            raise KeyError(
                f"Emotion '{emo}' not found in model encoder. "
                f"Known: {sorted(self.emotion_encoder.keys())}"
            )
        emotion_label = self.emotion_encoder[emo]

        return {
            "input_values": input_values,
            "emotion": torch.tensor(emotion_label, dtype=torch.long),
        }


def evaluate_emotion(model, loader: DataLoader, id2emotion: dict[int, str], device: torch.device):
    correct = 0
    total = 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for batch in loader:
            logits, _, _ = model(batch["input_values"].to(device))
            preds = logits.argmax(1)
            gts = batch["emotion"].to(device)

            correct_mask = preds.eq(gts)
            correct += int(correct_mask.sum().item())
            total += int(gts.size(0))

            for p, y, ok in zip(preds.cpu().tolist(), gts.cpu().tolist(), correct_mask.cpu().tolist()):
                y_lbl = id2emotion[int(y)]
                p_lbl = id2emotion[int(p)]
                per_class[y_lbl]["total"] += 1
                per_class[y_lbl]["correct"] += int(ok)
                confusion[y_lbl][p_lbl] += 1

    n = max(total, 1)
    per_class_acc = {
        k: (v["correct"] / max(v["total"], 1)) for k, v in sorted(per_class.items())
    }
    support = {k: int(v["total"]) for k, v in sorted(per_class.items())}

    confusion_out = {y: dict(sorted(preds.items())) for y, preds in sorted(confusion.items())}

    return {
        "n": total,
        "accuracy": correct / n,
        "per_class_accuracy": per_class_acc,
        "support": support,
        "confusion": confusion_out,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate our model on KazEmoTTS emotions only.")
    parser.add_argument("--split", default=None, help="Dataset split (default: first available).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint .pt (default: auto)")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "kazemotts_emotion_metrics.json"),
        help="Where to write metrics JSON.",
    )
    parser.add_argument("--cache-dir", default=None, help="Hugging Face datasets cache dir.")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt_path = Path(args.checkpoint) if args.checkpoint else resolve_checkpoint()
    (model, emotion_encoder, _, _, id2emotion, _, _, processor) = load_model(ckpt_path, device)

    ds = load_kazemotts(cache_dir=args.cache_dir)
    if args.split and args.split in ds:
        split_name = args.split
    else:
        split_name = next(iter(ds.keys()))
    split = ds[split_name]

    eval_ds = KazEmoEmotionDataset(split, processor, emotion_encoder)
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    res = evaluate_emotion(model, loader, id2emotion, device)
    out = {
        "dataset": "issai/KazEmoTTS",
        "split": split_name,
        "checkpoint": str(ckpt_path),
        "sample_rate": SAMPLE_RATE,
        "model_label_encoder_path": str(Path(MODEL_DIR) / "label_encoders.json"),
        "metrics": res,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

