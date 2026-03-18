#!/usr/bin/env python3
"""Train HuBERT with a single head per feature (emotion, gender, or age).

By default trains for emotion; use --feature age or --feature gender for the others.
Saves one model per feature under single_head/models/{feature}/.
"""

import argparse
import os
import signal
import sys
import warnings
import logging
from pathlib import Path

# Ensure repo root is on path when running from single_head/
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ['DATASETS_AUDIO_BACKEND'] = 'soundfile'
os.environ['TORCHCODEC_QUIET'] = '1'
os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '0'
# Reduce VRAM fragmentation from PyTorch's caching allocator.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='torch._dynamo')
warnings.filterwarnings('ignore', module='torch._inductor.utils')
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from datasets import Audio
import numpy as np
from tqdm import tqdm
import json
import math
import random

from load_data import load, read_audio

# Feature choices: one head per run
FEATURES = ("emotion", "age", "gender")
# Dataset column for age
AGE_COLUMN = "age_category"

BATCH_SIZE = 8
HEAD_LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
GRAD_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 42

if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    # benchmark=True spikes VRAM during cuDNN algorithm search. Inputs are
    # fixed-length (max_length padding), so benchmarking gives no benefit.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision('high')

_stop_requested = False


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set global seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, '_orig_mod', model)


def _sigint_handler(signum, frame):
    global _stop_requested
    if _stop_requested:
        print("\nSecond Ctrl+C: exiting immediately.", file=sys.stderr)
        sys.exit(130)
    _stop_requested = True
    print("\nCtrl+C received. Finishing current batch and saving checkpoint...")


def _weighted_pool(all_layers: torch.Tensor, layer_weights: torch.Tensor) -> torch.Tensor:
    w = torch.softmax(layer_weights, dim=0)
    pooled = (w.view(-1, 1, 1, 1) * all_layers).sum(dim=0)
    return pooled.mean(dim=1)


def _spec_augment(
    all_layers: torch.Tensor,
    time_mask_param: int = 70,
    freq_mask_param: int = 27,
    num_time_masks: int = 2,
    num_freq_masks: int = 2,
) -> torch.Tensor:
    """SpecAugment-style masking on stacked hidden states [num_layers, B, T, H].
    LB policy (Park et al. 2019): F=27, T=70, multiplicity 2."""
    _, _, T, H = all_layers.shape
    out = all_layers.clone()
    device = all_layers.device
    for _ in range(num_time_masks):
        t_len = min(time_mask_param, max(1, T - 1))
        t_start = torch.randint(0, max(1, T - t_len + 1), (1,), device=device).item()
        out[:, :, t_start : t_start + t_len, :] = 0.0
    for _ in range(num_freq_masks):
        f_len = min(freq_mask_param, max(1, H - 1))
        f_start = torch.randint(0, max(1, H - f_len + 1), (1,), device=device).item()
        out[:, :, :, f_start : f_start + f_len] = 0.0
    return out


def _make_cosine_schedule(
    optimizer: optim.Optimizer,
    hold_epochs: int,
    decay_epochs: int,
    eta_min_factor: float = 1e-2,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Hold LR for hold_epochs, then cosine-anneal to eta_min_factor * base_lr.

    Works for any number of param groups — each group's LR is scaled by the
    same lambda, so relative differences (e.g. layer-decay in finetune) are kept.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < hold_epochs:
            return 1.0
        # Use (decay_epochs - 1) so the minimum is reached at the very last
        # training epoch rather than one step after it.
        t = min((epoch - hold_epochs) / max(decay_epochs - 1, 1), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return eta_min_factor + (1.0 - eta_min_factor) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class SingleHeadHubert(nn.Module):
    """HuBERT with one head and one set of layer weights for a single feature."""

    def __init__(self, feature: str, num_classes: int, freeze_backbone: bool = True, use_spec_augment: bool = False):
        super().__init__()
        self.feature = feature
        self.num_classes = num_classes
        # Prefer the model's built-in SpecAugment (masks projected features pre-transformer)
        # over manually masking post-transformer hidden states.
        self.use_spec_augment = use_spec_augment
        self.hubert = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960", output_hidden_states=True
        )
        if hasattr(self.hubert, "config"):
            self.hubert.config.apply_spec_augment = bool(use_spec_augment)
            self.hubert.config.mask_time_prob = 0.05
            self.hubert.config.mask_time_length = 10
            self.hubert.config.mask_feature_prob = 0.0
            self.hubert.config.mask_feature_length = 10
        if hasattr(self.hubert.config, "training_drop_path"):
            self.hubert.config.training_drop_path = 0.1
        if freeze_backbone:
            for param in self.hubert.parameters():
                param.requires_grad = False

        hidden_size = 768
        num_layers = 13
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_values):
        outputs = self.hubert(input_values)
        backbone_frozen = not any(p.requires_grad for p in self.hubert.parameters())
        if backbone_frozen:
            all_layers = torch.stack(
                [h.detach().clone() for h in outputs.hidden_states]
            )
        else:
            all_layers = torch.stack(outputs.hidden_states, dim=0)
        feats = _weighted_pool(all_layers, self.layer_weights)
        return self.head(feats)


class AudioDataset(torch.utils.data.Dataset):
    """Dataset for audio + labels; same as multi-head but we only use one label in the loop."""

    def __init__(self, dataset_split, processor, emotion_encoder, gender_encoder, age_encoder,
                 max_length=160000):
        self.data = dataset_split
        self.processor = processor
        self.emotion_encoder = emotion_encoder
        self.gender_encoder = gender_encoder
        self.age_encoder = age_encoder
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        audio_data, sample_rate = read_audio(row['audio'])
        if sample_rate != SAMPLE_RATE:
            import librosa
            audio_data = librosa.resample(
                audio_data.astype(np.float32),
                orig_sr=sample_rate, target_sr=SAMPLE_RATE
            )
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        if len(audio_data) > self.max_length:
            audio_data = audio_data[:self.max_length]
        else:
            audio_data = np.pad(audio_data, (0, self.max_length - len(audio_data)))

        inputs = self.processor(
            audio_data, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.squeeze(0)

        def norm(v):
            return v if (v is not None and str(v).strip()) else "unknown"

        emotion_raw = row.get("emotion")
        emotion_normalized = norm(emotion_raw)
        gender_raw = row.get("gender")
        gender_normalized = norm(gender_raw)
        age_raw = row.get(AGE_COLUMN)
        age_normalized = norm(age_raw)

        enc = self.emotion_encoder
        emotion_label = enc.get(emotion_normalized, enc.get("unknown", 0))
        enc = self.gender_encoder
        gender_label = enc.get(gender_normalized, enc.get("unknown", 0))
        enc = self.age_encoder
        age_label = enc.get(age_normalized, enc.get("unknown", 0))

        return {
            "input_values": input_values,
            "emotion": torch.tensor(emotion_label, dtype=torch.long),
            "gender": torch.tensor(gender_label, dtype=torch.long),
            "age": torch.tensor(age_label, dtype=torch.long),
        }


def build_label_encoders(dataset):
    """Build emotion, gender, age encoders (same as train.py)."""
    emotions = set()
    genders = set()
    age_categories = set()
    has_missing_emotion = False
    has_missing_gender = False
    has_missing_age = False

    for split_name in dataset.keys():
        split = dataset[split_name]
        split_no_audio = split.remove_columns(["audio"]) if "audio" in split.column_names else split
        for row in split_no_audio:
            if "emotion" in row and row["emotion"] is not None and str(row["emotion"]).strip():
                emotions.add(str(row["emotion"]).strip())
            else:
                has_missing_emotion = True
            if "gender" in row and row["gender"] is not None and str(row["gender"]).strip():
                genders.add(str(row["gender"]).strip())
            else:
                has_missing_gender = True
            key = AGE_COLUMN
            if key in row and row[key] is not None and str(row[key]).strip():
                age_categories.add(str(row[key]).strip())
            else:
                has_missing_age = True

    emotion_encoder = {label: idx for idx, label in enumerate(sorted(emotions))}
    gender_encoder = {label: idx for idx, label in enumerate(sorted(genders))}
    age_encoder = {label: idx for idx, label in enumerate(sorted(age_categories))}
    if has_missing_emotion:
        emotion_encoder["unknown"] = len(emotion_encoder)
    if has_missing_gender:
        gender_encoder["unknown"] = len(gender_encoder)
    if has_missing_age:
        age_encoder["unknown"] = len(age_encoder)
    return emotion_encoder, gender_encoder, age_encoder


def train_epoch(model, dataloader, criterion, optimizer, device, feature: str):
    global _stop_requested
    model.train()
    total_loss = 0.0
    num_batches = 0
    _use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if _use_bf16 else torch.float32

    for batch in tqdm(dataloader, desc="Training"):
        if _stop_requested:
            break
        input_values = batch["input_values"].to(device, non_blocking=True)
        labels = batch[feature].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            logits = model(input_values)
            loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return {"total": total_loss / max(num_batches, 1)}, _stop_requested


def validate(model, dataloader, criterion, device, feature: str):
    global _stop_requested
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    num_batches = 0
    _use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if _use_bf16 else torch.float32
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if _stop_requested:
                break
            input_values = batch["input_values"].to(device, non_blocking=True)
            labels = batch[feature].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(input_values)
                loss = criterion(logits, labels)
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total_samples += labels.size(0)
            num_batches += 1
    n = max(total_samples, 1)
    return {
        "total": total_loss / max(num_batches, 1),
        "acc": correct / n,
    }


def main():
    global _stop_requested
    parser = argparse.ArgumentParser(description="Train single-head HuBERT (one feature per model).")
    parser.add_argument(
        "--feature",
        default="emotion",
        choices=list(FEATURES),
        help="Target feature to train (default: emotion).",
    )
    args = parser.parse_args()
    feature = args.feature

    set_seed(RANDOM_SEED)
    signal.signal(signal.SIGINT, _sigint_handler)
    print(f"Using device: {DEVICE}")
    print(f"Training single-head model for feature: {feature}")
    print("Press Ctrl+C to stop training and save a checkpoint.")

    # Per-feature model dir: single_head/models/emotion | age | gender
    MODEL_BASE = Path(__file__).resolve().parent / "models"
    MODEL_DIR = MODEL_BASE / feature
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load()
    print("Building label encoders...")
    emotion_encoder, gender_encoder, age_encoder = build_label_encoders(dataset)
    encoders = {"emotion": emotion_encoder, "gender": gender_encoder, "age": age_encoder}
    encoder = encoders[feature]
    num_classes = len(encoder)
    print(f"Found {num_classes} classes for {feature}: {list(encoder.keys())}")

    encoder_path = MODEL_DIR / "label_encoders.json"
    with open(encoder_path, "w") as f:
        json.dump({feature: encoder}, f, indent=2)
    print(f"Saved label encoders to {encoder_path}")

    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    train_split = dataset.get("train") or dataset[list(dataset.keys())[0]]
    val_split = dataset.get("validation") or dataset.get("val") or dataset.get("test")
    if "audio" in train_split.column_names:
        train_split = train_split.cast_column("audio", Audio(decode=False))
    if val_split is not None and "audio" in val_split.column_names:
        val_split = val_split.cast_column("audio", Audio(decode=False))

    train_dataset = AudioDataset(
        train_split, processor,
        emotion_encoder, gender_encoder, age_encoder,
    )
    val_dataset = AudioDataset(
        val_split or train_split, processor,
        emotion_encoder, gender_encoder, age_encoder,
    )

    _pin = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=_pin,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=_pin,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print("Initializing model...")
    model = SingleHeadHubert(
        feature=feature,
        num_classes=num_classes,
        freeze_backbone=True,
    ).to(DEVICE)

    head_params = list(model.head.parameters()) + [model.layer_weights]
    _fused_opt = DEVICE.type == "cuda"
    try:
        optimizer = optim.AdamW(
            [{"params": head_params, "lr": HEAD_LEARNING_RATE}],
            weight_decay=0.01, fused=_fused_opt
        )
        if _fused_opt:
            print("  Optimizer: fused AdamW")
    except (TypeError, RuntimeError):
        optimizer = optim.AdamW(
            [{"params": head_params, "lr": HEAD_LEARNING_RATE}],
            weight_decay=0.01,
        )
    # LR schedule: hold HEAD_LEARNING_RATE for 3 epochs, then cosine decay for 7 epochs
    # to 1e-2 × initial LR  (1e-3 → 1e-5)
    scheduler = _make_cosine_schedule(optimizer, hold_epochs=3, decay_epochs=7)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    metrics_path = MODEL_DIR / "training_metrics.json"
    latest_path = MODEL_DIR / "latest_checkpoint.pt"
    best_val_loss = float("inf")
    last_epoch = -1
    all_metrics = []
    start_epoch = 0
    epochs_without_improvement = 0

    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=DEVICE, weights_only=False)
        if (ckpt.get("feature") == feature and ckpt.get("num_classes") == num_classes
                and ckpt["epoch"] + 1 < NUM_EPOCHS):
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            start_epoch = ckpt["epoch"] + 1
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            else:
                for _ in range(start_epoch):
                    scheduler.step()
            if metrics_path.exists():
                with open(metrics_path) as f:
                    all_metrics = json.load(f)
            print(f"Resuming from epoch {start_epoch + 1}/{NUM_EPOCHS} (latest checkpoint)")

    if DEVICE.type == "cuda" and hasattr(torch, "compile"):
        try:
            print("Compiling model with torch.compile(mode='default')...")
            model = torch.compile(model, mode="default", dynamic=None)
        except Exception as e:
            print(f"  torch.compile unavailable ({e}), running eager.")

    for epoch in range(start_epoch, NUM_EPOCHS):
        if _stop_requested:
            break
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} [{feature}]")
        print("-" * 50)
        train_metrics, stopped = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, feature
        )
        last_epoch = epoch
        print(f"Train Loss: {train_metrics['total']:.4f}")
        if stopped:
            break
        val_metrics = validate(model, val_loader, criterion, DEVICE, feature)
        if _stop_requested:
            break
        print(f"Val Loss: {val_metrics['total']:.4f}  Val Acc: {val_metrics['acc']:.4f}")

        def layer_prefs_tolist(weights):
            return torch.softmax(weights, dim=0).detach().cpu().tolist()

        _m = _unwrap(model)
        epoch_record = {
            "epoch": epoch + 1,
            "train": {k: round(float(v), 6) for k, v in train_metrics.items()},
            "val": {k: round(float(v), 6) for k, v in val_metrics.items()},
            "layer_prefs": {feature: [round(x, 6) for x in layer_prefs_tolist(_m.layer_weights)]},
        }
        all_metrics.append(epoch_record)
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            epochs_without_improvement = 0
            model_path = MODEL_DIR / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": _unwrap(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "feature": feature,
                "num_classes": num_classes,
            }, model_path)
            print(f"Saved best model to {model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping: no val loss improvement for {EARLY_STOPPING_PATIENCE} epochs.")
                break

        # Advance LR schedule and log current LR
        scheduler.step()
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": _unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "feature": feature,
            "num_classes": num_classes,
        }, latest_path)

    if _stop_requested:
        checkpoint_path = MODEL_DIR / "checkpoint_interrupted.pt"
        torch.save({
            "epoch": last_epoch,
            "model_state_dict": _unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "feature": feature,
            "num_classes": num_classes,
        }, checkpoint_path)
        print(f"\nStopped by user. Checkpoint saved to {checkpoint_path}")
    else:
        print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
