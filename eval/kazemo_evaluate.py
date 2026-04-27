#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["TORCHCODEC_QUIET"] = "1"

# Allow running as `python eval/kazemo_evaluate.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eval.inference import load_model, resolve_checkpoint
from eval.results_dir import RESULTS_DIR
from loaders.load_data import read_audio
from multihead.utils import MODEL_DIR, SAMPLE_RATE

from loaders.kazemo.load_data import load_kazemotts


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m {secs}s"
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _estimate_training_span_from_files(ckpt_path: Path) -> tuple[datetime, datetime] | None:
    if "finetune" in ckpt_path.parts:
        run_dir = ckpt_path.parent
        latest = run_dir / "latest_checkpoint_finetune.pt"
    else:
        run_dir = ckpt_path.parent
        latest = run_dir / "latest_checkpoint.pt"

    candidates = [ckpt_path]
    if latest.exists():
        candidates.append(latest)

    step_dir = run_dir / "steps"
    if step_dir.exists():
        candidates.extend(step_dir.glob("step_*.pt"))

    if not candidates:
        return None

    mtimes = []
    for p in candidates:
        try:
            mtimes.append((Path(p), Path(p).stat().st_mtime))
        except OSError:
            continue

    if not mtimes:
        return None

    _, start_ts = min(mtimes, key=lambda x: x[1])
    _, end_ts = max(mtimes, key=lambda x: x[1])
    return (datetime.fromtimestamp(start_ts), datetime.fromtimestamp(end_ts))


def print_training_details(ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch_idx = ckpt.get("epoch")
    trained_epochs = int(epoch_idx) + 1 if epoch_idx is not None else None
    global_step = ckpt.get("global_step")
    samples_seen = ckpt.get("samples_seen")
    val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss"))
    num_emotions = ckpt.get("num_emotions")
    num_genders = ckpt.get("num_genders")
    num_ages = ckpt.get("num_ages")
    unfrozen_layers = ckpt.get("unfrozen_layers")

    stage = "stage-2 finetune" if "finetune" in ckpt_path.parts else "stage-1 training"
    print(f"Training details ({stage}):", flush=True)
    print(f"  checkpoint: {ckpt_path}", flush=True)
    if trained_epochs is not None:
        print(f"  trained_epochs: {trained_epochs}", flush=True)
    if global_step is not None:
        print(f"  global_steps: {global_step}", flush=True)
    if samples_seen is not None:
        print(f"  samples_seen: {samples_seen}", flush=True)
    if val_loss is not None:
        print(f"  val_loss: {float(val_loss):.6f}", flush=True)
    if num_emotions is not None and num_genders is not None and num_ages is not None:
        print(
            f"  classes: emotions={num_emotions}, gender={num_genders}, age={num_ages}",
            flush=True,
        )
    if unfrozen_layers:
        print(f"  unfrozen_layers: {sorted(unfrozen_layers)}", flush=True)

    span = _estimate_training_span_from_files(ckpt_path)
    if span is not None:
        start_dt, end_dt = span
        duration = end_dt - start_dt
        print(
            "  training_wall_time_estimate: "
            f"{_format_duration(duration.total_seconds())} "
            f"(from file mtimes: {start_dt.isoformat(timespec='seconds')} -> "
            f"{end_dt.isoformat(timespec='seconds')})",
            flush=True,
        )
    else:
        print("  training_wall_time_estimate: unavailable", flush=True)


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
        # Match demo.py behavior: ensure float32 waveform before padding/processor.
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if len(audio_data) > self.max_length:
            audio_data = audio_data[: self.max_length]
        raw_length = len(audio_data)
        if raw_length < self.max_length:
            audio_data = np.pad(audio_data, (0, self.max_length - raw_length))

        inputs = self.processor(
            audio_data, sampling_rate=SAMPLE_RATE, return_tensors="pt",
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
            "input_length": torch.tensor(raw_length, dtype=torch.long),
            "emotion": torch.tensor(emotion_label, dtype=torch.long),
        }


def evaluate_emotion(model, loader: DataLoader, id2emotion: dict[int, str], device: torch.device):
    correct = 0
    top2_correct = 0
    total = 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            _iv = batch["input_values"].to(device)
            _il = batch.get("input_length")
            if _il is not None:
                _il = _il.to(device)
            logits, _, _ = model(_iv, input_lengths=_il)
            preds = logits.argmax(1)
            top2 = logits.topk(k=min(2, logits.size(-1)), dim=1).indices
            gts = batch["emotion"].to(device)

            correct_mask = preds.eq(gts)
            correct += int(correct_mask.sum().item())
            top2_correct += int(top2.eq(gts.unsqueeze(1)).any(dim=1).sum().item())
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
        "top2_accuracy": top2_correct / n,
        "per_class_accuracy": per_class_acc,
        "support": support,
        "confusion": confusion_out,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate our model on KazEmoTTS emotions only.")
    parser.add_argument("--split", default=None, help="Dataset split (default: first available).")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to first N samples (useful for quick checks).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint .pt (default: auto)")
    parser.add_argument(
        "--out",
        default=str(RESULTS_DIR / "kazemotts_emotion_metrics.json"),
        help="Where to write metrics JSON (default: results/kazemotts_emotion_metrics.json).",
    )
    parser.add_argument("--cache-dir", default=None, help="Hugging Face datasets cache dir.")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}", flush=True)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else resolve_checkpoint()
    print_training_details(ckpt_path)
    (model, emotion_encoder, _, _, id2emotion, _, _, processor) = load_model(ckpt_path, device)
    print(f"Checkpoint: {ckpt_path}", flush=True)

    ds = load_kazemotts(cache_dir=args.cache_dir, max_samples=args.max_samples)
    if args.split and args.split in ds:
        split_name = args.split
    else:
        # Prefer an eval-ish split name when present; fall back to first.
        for candidate in ("test", "validation", "val", "train"):
            if candidate in ds:
                split_name = candidate
                break
        else:
            split_name = next(iter(ds.keys()))
    split = ds[split_name]
    if args.max_samples is not None and args.max_samples > 0 and len(split) > args.max_samples:
        split = split.select(range(args.max_samples))
    print(f"Split: {split_name} | n={len(split)}", flush=True)
    print("Starting evaluation...", flush=True)

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
    print(f"Wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()

