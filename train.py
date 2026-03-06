#!/usr/bin/env python3
"""Fine-tune HuBERT with multiple heads (emotion, gender, age) on audio data."""

import os
import signal
import sys
import warnings
import logging

# Disable torchcodec and use soundfile for audio decoding
os.environ['DATASETS_AUDIO_BACKEND'] = 'soundfile'
os.environ['TORCHCODEC_QUIET'] = '1'
# Uncomment to silence "Not enough SMs to use max_autotune_gemm" on smaller GPUs:
os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '0'
# Reduce VRAM fragmentation from PyTorch's caching allocator (especially after
# interrupted runs). expandable_segments lets the allocator release and reuse
# smaller blocks instead of holding onto large reserved chunks.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
warnings.filterwarnings('ignore', category=UserWarning)
# Suppress torch.compile recompile/duplicate-tensor and inductor max_autotune messages
warnings.filterwarnings('ignore', module='torch._dynamo')
warnings.filterwarnings('ignore', module='torch._inductor.utils')
# Suppress dynamo logger in case it uses logging for the same messages
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from datasets import Audio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import math

# Import load_data (warnings already suppressed in load_data.py)
from load_data import load, read_audio, DATA_DIR

# Configuration
BATCH_SIZE = 8
HEAD_LEARNING_RATE = 1e-3  # Higher LR for new heads
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000  # HuBERT requires 16kHz

# ── GPU throughput optimizations (PyTorch 2.x / CUDA 12.x) ──────────────────
if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32 matmul (disabled by default)
    # benchmark=True probes cuDNN conv algorithms on first use, spiking VRAM by
    # ~0.5-1.5 GB. Since all inputs are padded to a fixed max_length the optimal
    # algorithm never changes between runs, so the benchmark overhead is pure waste.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False      # still allow non-deterministic (fast) kernels
    torch.set_float32_matmul_precision('high')      # prefer TF32 over full fp32

# Loss weights for multi-task learning
EMOTION_WEIGHT = 1.2
GENDER_WEIGHT = 0.5
AGE_WEIGHT = 1.0

GRAD_CLIP_NORM = 1.0      # max gradient norm for clipping (stabilises mixed-precision training)

# Model save directory
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Safe stop: set by SIGINT handler (Ctrl+C)
_stop_requested = False


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the original module, unwrapping torch.compile's OptimizedModule if present.

    torch.compile wraps the model in OptimizedModule, prefixing all state_dict
    keys with '_orig_mod.'. Always save/load checkpoints via _unwrap() so that
    saved weights are portable (loadable without torch.compile).
    """
    return getattr(model, '_orig_mod', model)


def _sigint_handler(signum, frame):
    global _stop_requested
    if _stop_requested:
        print("\nSecond Ctrl+C: exiting immediately.", file=sys.stderr)
        sys.exit(130)
    _stop_requested = True
    print("\nCtrl+C received. Finishing current batch and saving checkpoint...")


def _weighted_pool(all_layers: torch.Tensor, layer_weights: torch.Tensor) -> torch.Tensor:
    """Softmax-weighted sum over HuBERT hidden layers, then mean-pool over time.

    Args:
        all_layers:    [num_layers, B, T, H]  – stacked hidden states
        layer_weights: [num_layers]           – learnable unnormalised weights
    Returns:
        [B, H] pooled features
    """
    w = torch.softmax(layer_weights, dim=0)
    pooled = (w.view(-1, 1, 1, 1) * all_layers).sum(dim=0)  # [B, T, H]
    return pooled.mean(dim=1)                                # [B, H]


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


class MultiTaskHubert(nn.Module):
    """HuBERT model with three heads using task-specific weighted layer sums."""

    def __init__(self, num_emotions, num_genders, num_ages, freeze_backbone=True):
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=True,)

        if freeze_backbone:
            for param in self.hubert.parameters():
                param.requires_grad = False

        hidden_size = 768
        num_layers = 13  # 12 transformer layers + initial embedding layer

        # Learnable per-task layer weights (before softmax)
        self.emotion_weights = nn.Parameter(torch.ones(num_layers))
        self.gender_weights = nn.Parameter(torch.ones(num_layers))
        self.age_weights = nn.Parameter(torch.ones(num_layers))

        # Slightly deeper heads with non-linearity
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_emotions),
        )
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_genders),
        )
        self.age_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_ages),
        )

    def forward(self, input_values):
        """
        Args:
            input_values: Audio features [batch_size, sequence_length]
        Returns:
            emotion_logits, gender_logits, age_logits
        """
        outputs = self.hubert(input_values)

        # hidden_states: tuple of 13 tensors [batch, time, hidden].
        # When backbone is fully frozen: .detach().clone() saves memory and breaks
        # HuggingFace storage aliasing so dynamo sees 13 independent tensors.
        # When any layer is unfrozen (finetune): do not detach so gradients flow.
        backbone_frozen = not any(p.requires_grad for p in self.hubert.parameters())
        if backbone_frozen:
            all_layers = torch.stack(
                [h.detach().clone() for h in outputs.hidden_states]
            )  # [13, B, T, H]
        else:
            all_layers = torch.stack(outputs.hidden_states, dim=0)

        emo_feats = _weighted_pool(all_layers, self.emotion_weights)
        gen_feats = _weighted_pool(all_layers, self.gender_weights)
        age_feats = _weighted_pool(all_layers, self.age_weights)

        return (
            self.emotion_head(emo_feats),
            self.gender_head(gen_feats),
            self.age_head(age_feats),
        )


class AudioDataset(Dataset):
    """Dataset for audio classification with emotion, gender, and age_category labels."""

    def __init__(self, dataset_split, processor, emotion_encoder, gender_encoder, age_encoder,
                 max_length=160000):  # 10 seconds at 16kHz
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
        
        # Read audio
        audio_data, sample_rate = read_audio(row['audio'])
        
        # Resample to 16kHz if needed
        if sample_rate != SAMPLE_RATE:
            import librosa
            audio_data = librosa.resample(
                audio_data.astype(np.float32),
                orig_sr=sample_rate,
                target_sr=SAMPLE_RATE
            )
        
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Truncate or pad to max_length
        if len(audio_data) > self.max_length:
            audio_data = audio_data[:self.max_length]
        else:
            audio_data = np.pad(audio_data, (0, self.max_length - len(audio_data)))
        
        # Process with HuBERT processor
        inputs = self.processor(
            audio_data,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values.squeeze(0)
        
        # Get labels
        emotion_raw = row.get('emotion')
        emotion_normalized = emotion_raw if (emotion_raw is not None and str(emotion_raw).strip()) else 'unknown'

        gender_raw = row.get('gender')
        gender_normalized = gender_raw if (gender_raw is not None and str(gender_raw).strip()) else 'unknown'

        age_raw = row.get('age_category')
        age_normalized = age_raw if (age_raw is not None and str(age_raw).strip()) else 'unknown'

        if 'unknown' in self.emotion_encoder:
            emotion_label = self.emotion_encoder.get(emotion_normalized, self.emotion_encoder['unknown'])
        else:
            emotion_label = self.emotion_encoder[emotion_normalized]

        if 'unknown' in self.gender_encoder:
            gender_label = self.gender_encoder.get(gender_normalized, self.gender_encoder['unknown'])
        else:
            gender_label = self.gender_encoder[gender_normalized]

        if 'unknown' in self.age_encoder:
            age_label = self.age_encoder.get(age_normalized, self.age_encoder['unknown'])
        else:
            age_label = self.age_encoder[age_normalized]

        return {
            'input_values': input_values,
            'emotion': torch.tensor(emotion_label, dtype=torch.long),
            'gender': torch.tensor(gender_label, dtype=torch.long),
            'age': torch.tensor(age_label, dtype=torch.long)
        }


def build_label_encoders(dataset):
    """Build label encoders for emotion, gender, and age_category from the dataset.
    Adds 'unknown' only if the dataset contains missing/empty labels."""
    emotions = set()
    genders = set()
    age_categories = set()
    has_missing_emotion = False
    has_missing_gender = False
    has_missing_age = False

    for split_name in dataset.keys():
        split = dataset[split_name]
        split_no_audio = split.remove_columns(['audio']) if 'audio' in split.column_names else split
        for row in split_no_audio:
            if 'emotion' in row and row['emotion'] is not None and str(row['emotion']).strip():
                emotions.add(str(row['emotion']).strip())
            else:
                has_missing_emotion = True
            if 'gender' in row and row['gender'] is not None and str(row['gender']).strip():
                genders.add(str(row['gender']).strip())
            else:
                has_missing_gender = True
            if 'age_category' in row and row['age_category'] is not None and str(row['age_category']).strip():
                age_categories.add(str(row['age_category']).strip())
            else:
                has_missing_age = True

    emotion_encoder = {label: idx for idx, label in enumerate(sorted(emotions))}
    gender_encoder = {label: idx for idx, label in enumerate(sorted(genders))}
    age_encoder = {label: idx for idx, label in enumerate(sorted(age_categories))}

    if has_missing_emotion:
        emotion_encoder['unknown'] = len(emotion_encoder)
    if has_missing_gender:
        gender_encoder['unknown'] = len(gender_encoder)
    if has_missing_age:
        age_encoder['unknown'] = len(age_encoder)

    return emotion_encoder, gender_encoder, age_encoder


def train_epoch(model, dataloader, criterion_emotion, criterion_gender,
                criterion_age, optimizer, device):
    """Train for one epoch. Returns (metrics_dict, was_stopped)."""
    global _stop_requested
    model.train()
    total_loss = 0.0
    emotion_loss_sum = 0.0
    gender_loss_sum = 0.0
    age_loss_sum = 0.0
    num_batches = 0

    # BF16 autocast — only when the GPU actually supports it
    _use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if _use_bf16 else torch.float32

    for batch in tqdm(dataloader, desc="Training"):
        if _stop_requested:
            break
        # non_blocking=True overlaps H→D transfer with GPU work
        input_values = batch['input_values'].to(device, non_blocking=True)
        emotion_labels = batch['emotion'].to(device, non_blocking=True)
        gender_labels = batch['gender'].to(device, non_blocking=True)
        age_labels = batch['age'].to(device, non_blocking=True)

        # set_to_none avoids a memset; faster than zeroing
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            emotion_logits, gender_logits, age_logits = model(input_values)

            # Compute losses
            loss_emotion = criterion_emotion(emotion_logits, emotion_labels)
            loss_gender = criterion_gender(gender_logits, gender_labels)
            loss_age = criterion_age(age_logits, age_labels)

            # Combined loss
            total_batch_loss = (
                EMOTION_WEIGHT * loss_emotion +
                GENDER_WEIGHT * loss_gender +
                AGE_WEIGHT * loss_age
            )

        total_batch_loss.backward()
        # Clip gradients before step (guards against spikes in unfrozen layers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += total_batch_loss.item()
        emotion_loss_sum += loss_emotion.item()
        gender_loss_sum += loss_gender.item()
        age_loss_sum += loss_age.item()
        num_batches += 1

    if num_batches == 0:
        num_batches = 1
    return {
        'total': total_loss / num_batches,
        'emotion': emotion_loss_sum / num_batches,
        'gender': gender_loss_sum / num_batches,
        'age': age_loss_sum / num_batches
    }, _stop_requested


def validate(model, dataloader, criterion_emotion, criterion_gender, 
             criterion_age, device):
    """Validate the model."""
    global _stop_requested
    model.eval()
    total_loss = 0.0
    emotion_loss_sum = 0.0
    gender_loss_sum = 0.0
    age_loss_sum = 0.0
    
    emotion_correct = 0
    gender_correct = 0
    age_correct = 0
    total_samples = 0
    num_batches = 0

    _use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if _use_bf16 else torch.float32
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if _stop_requested:
                break
            input_values = batch['input_values'].to(device, non_blocking=True)
            emotion_labels = batch['emotion'].to(device, non_blocking=True)
            gender_labels = batch['gender'].to(device, non_blocking=True)
            age_labels = batch['age'].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                emotion_logits, gender_logits, age_logits = model(input_values)

                # Compute losses
                loss_emotion = criterion_emotion(emotion_logits, emotion_labels)
                loss_gender = criterion_gender(gender_logits, gender_labels)
                loss_age = criterion_age(age_logits, age_labels)

                total_batch_loss = (
                    EMOTION_WEIGHT * loss_emotion +
                    GENDER_WEIGHT * loss_gender +
                    AGE_WEIGHT * loss_age
                )

            total_loss += total_batch_loss.item()
            emotion_loss_sum += loss_emotion.item()
            gender_loss_sum += loss_gender.item()
            age_loss_sum += loss_age.item()

            # Compute accuracy
            emotion_pred = torch.argmax(emotion_logits, dim=1)
            gender_pred = torch.argmax(gender_logits, dim=1)
            age_pred = torch.argmax(age_logits, dim=1)

            emotion_correct += (emotion_pred == emotion_labels).sum().item()
            gender_correct += (gender_pred == gender_labels).sum().item()
            age_correct += (age_pred == age_labels).sum().item()
            total_samples += emotion_labels.size(0)
            num_batches += 1

    if num_batches == 0:
        num_batches = 1
    if total_samples == 0:
        total_samples = 1
    return {
        'total': total_loss / num_batches,
        'emotion': emotion_loss_sum / num_batches,
        'gender': gender_loss_sum / num_batches,
        'age': age_loss_sum / num_batches,
        'emotion_acc': emotion_correct / total_samples,
        'gender_acc': gender_correct / total_samples,
        'age_acc': age_correct / total_samples
    }


def main():
    global _stop_requested
    signal.signal(signal.SIGINT, _sigint_handler)
    print(f"Using device: {DEVICE}")
    print("Press Ctrl+C to stop training and save a checkpoint.")

    # Load dataset
    dataset = load()
    
    # Build label encoders
    print("Building label encoders...")
    emotion_encoder, gender_encoder, age_encoder = build_label_encoders(dataset)
    num_emotions = len(emotion_encoder)
    num_genders = len(gender_encoder)
    num_ages = len(age_encoder)

    print(f"Found {num_emotions} emotion classes: {list(emotion_encoder.keys())}")
    print(f"Found {num_genders} gender classes: {list(gender_encoder.keys())}")
    print(f"Found {num_ages} age classes: {list(age_encoder.keys())}")

    # Save encoders
    encoder_path = MODEL_DIR / "label_encoders.json"
    with open(encoder_path, 'w') as f:
        json.dump({
            'emotion': emotion_encoder,
            'gender': gender_encoder,
            'age': age_encoder
        }, f, indent=2)
    print(f"Saved label encoders to {encoder_path}")
    
    # Initialize feature extractor
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    
    # Create datasets (disable automatic audio decoding)
    train_split = dataset.get('train', dataset.get('train', None))
    if train_split is None:
        # If no train split, use the first available split
        train_split = dataset[list(dataset.keys())[0]]
    
    # Disable automatic audio decoding (we handle it manually with read_audio)
    if 'audio' in train_split.column_names:
        train_split = train_split.cast_column('audio', Audio(decode=False))
    
    val_split = dataset.get('validation', dataset.get('val', dataset.get('test', None)))
    if val_split is not None and 'audio' in val_split.column_names:
        val_split = val_split.cast_column('audio', Audio(decode=False))
    
    train_dataset = AudioDataset(
        train_split,
        processor,
        emotion_encoder,
        gender_encoder,
        age_encoder
    )

    if val_split is not None:
        val_dataset = AudioDataset(
            val_split,
            processor,
            emotion_encoder,
            gender_encoder,
            age_encoder
        )
    else:
        val_dataset = None
        print("Warning: No validation split found. Using train split for validation.")
        val_dataset = train_dataset
    
    # Create data loaders
    # persistent_workers keeps worker processes alive between epochs (avoids fork overhead)
    # prefetch_factor=2 queues 2 batches per worker in advance
    _pin = DEVICE.type == 'cuda'
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
    
    # Initialize model
    print("Initializing model...")
    model = MultiTaskHubert(
        num_emotions=num_emotions,
        num_genders=num_genders,
        num_ages=num_ages,
        freeze_backbone=True
    ).to(DEVICE)

    # Collect trainable params
    backbone_params = list(model.hubert.parameters())
    head_params = (list(model.emotion_head.parameters()) +
                   list(model.gender_head.parameters()) +
                   list(model.age_head.parameters()) +
                   [model.emotion_weights, model.gender_weights, model.age_weights])

    # fused=True merges the optimizer loop into a single CUDA kernel
    _fused_opt = DEVICE.type == 'cuda'
    try:
        optimizer = optim.AdamW([
            {'params': head_params, 'lr': HEAD_LEARNING_RATE}
        ], weight_decay=0.01, fused=_fused_opt)
        if _fused_opt:
            print("  Optimizer: fused AdamW")
    except (TypeError, RuntimeError):
        optimizer = optim.AdamW([
            {'params': head_params, 'lr': HEAD_LEARNING_RATE}
        ], weight_decay=0.01)
        print("  Optimizer: standard AdamW (fused not available)")

    # LR schedule: hold HEAD_LEARNING_RATE for 3 epochs, then cosine decay for 7 epochs
    # to 1e-2 × initial LR  (1e-3 → 1e-5)
    scheduler = _make_cosine_schedule(optimizer, hold_epochs=3, decay_epochs=7)

    # Loss functions (no class weights; dataset is balanced)
    criterion_emotion = nn.CrossEntropyLoss()
    criterion_gender = nn.CrossEntropyLoss()
    criterion_age = nn.CrossEntropyLoss()

    # Resume from latest checkpoint if training didn't finish
    metrics_path = MODEL_DIR / "training_metrics.json"
    latest_path = MODEL_DIR / "latest_checkpoint.pt"
    best_val_loss = float('inf')
    last_epoch = -1
    all_metrics = []
    start_epoch = 0

    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=DEVICE, weights_only=False)
        if (ckpt.get("num_emotions") == num_emotions and ckpt.get("num_genders") == num_genders
                and ckpt.get("num_ages") == num_ages
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

    # torch.compile with mode='default' uses TorchInductor kernel fusion.
    # Graph breaks (e.g. from HuBERT's masking ops) are handled gracefully in
    # default mode — dynamo compiles the parts it can and runs the rest eagerly.
    # dynamic=None allows dynamic shapes after first compile to reduce recompiles.
    if DEVICE.type == 'cuda' and hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile(mode='default')...")
            model = torch.compile(model, mode='default', dynamic=None)
        except Exception as e:
            print(f"  torch.compile unavailable ({e}), running eager.")

    for epoch in range(start_epoch, NUM_EPOCHS):
        if _stop_requested:
            break
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)

        train_metrics, stopped = train_epoch(
            model, train_loader, criterion_emotion, criterion_gender,
            criterion_age, optimizer, DEVICE
        )
        last_epoch = epoch

        print(f"Train Loss - Total: {train_metrics['total']:.4f}, "
              f"Emotion: {train_metrics['emotion']:.4f}, "
              f"Gender: {train_metrics['gender']:.4f}, "
              f"Age: {train_metrics['age']:.4f}")
        if stopped:
            break

        # Validate
        val_metrics = validate(
            model, val_loader, criterion_emotion, criterion_gender,
            criterion_age, DEVICE
        )
        if _stop_requested:
            break

        print(f"Val Loss - Total: {val_metrics['total']:.4f}, "
              f"Emotion: {val_metrics['emotion']:.4f}, "
              f"Gender: {val_metrics['gender']:.4f}, "
              f"Age: {val_metrics['age']:.4f}")
        print(f"Val Metrics - Emotion Acc: {val_metrics['emotion_acc']:.4f}, "
              f"Gender Acc: {val_metrics['gender_acc']:.4f}, "
              f"Age Acc: {val_metrics['age_acc']:.4f}")

        # Save metrics for this epoch (JSON-serializable floats)
        def layer_prefs_tolist(weights):
            return torch.softmax(weights, dim=0).detach().cpu().tolist()

        epoch_record = {
            "epoch": epoch + 1,
            "train": {k: round(float(v), 6) for k, v in train_metrics.items()},
            "val": {k: round(float(v), 6) for k, v in val_metrics.items()},
            "layer_prefs": {
                "emotion": [round(x, 6) for x in layer_prefs_tolist(_unwrap(model).emotion_weights)],
                "gender": [round(x, 6) for x in layer_prefs_tolist(_unwrap(model).gender_weights)],
                "age": [round(x, 6) for x in layer_prefs_tolist(_unwrap(model).age_weights)],
            },
        }
        all_metrics.append(epoch_record)
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")

        # Save best model
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            model_path = MODEL_DIR / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': _unwrap(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'num_emotions': num_emotions,
                'num_genders': num_genders,
                'num_ages': num_ages,
            }, model_path)
            print(f"Saved best model to {model_path}")

        # Advance LR schedule and log current LR
        scheduler.step()
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save latest checkpoint (for resume; metadata: epoch = completed epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': _unwrap(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'num_emotions': num_emotions,
            'num_genders': num_genders,
            'num_ages': num_ages,
        }, latest_path)

    if _stop_requested:
        checkpoint_path = MODEL_DIR / "checkpoint_interrupted.pt"
        torch.save({
            'epoch': last_epoch,
            'model_state_dict': _unwrap(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'num_emotions': num_emotions,
            'num_genders': num_genders,
            'num_ages': num_ages,
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
