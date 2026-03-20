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
from datasets import Audio, DatasetDict, Features, Value, concatenate_datasets
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import math
import random

# Import load_data (warnings already suppressed in load_data.py)
from load_data import load, read_audio, DATA_DIR
from kazemo.load_data import load_kazemotts

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
EARLY_STOPPING_PATIENCE = 5  # stop when val loss does not improve for this many epochs

# Model save directory
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Safe stop: set by SIGINT handler (Ctrl+C)
_stop_requested = False

RANDOM_SEED = 42
VAL_FRACTION = 0.1
KAZEMO_MAX_SAMPLES = int(os.environ.get("KAZEMO_MAX_SAMPLES", "20000"))
KAZEMO_VAL_FRACTION = float(os.environ.get("KAZEMO_VAL_FRACTION", "0.1"))
USE_KAZEMO = os.environ.get("USE_KAZEMO", "1").strip().lower() not in {"0", "false", "no"}


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set global seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fallback_split_train_val(train_split, *, seed: int = RANDOM_SEED, val_fraction: float = VAL_FRACTION):
    """Create a disjoint (train, val) split from a single HF Dataset."""
    if train_split is None:
        raise ValueError("train_split must not be None")
    if len(train_split) < 2:
        raise ValueError(f"Not enough rows to split train/val (n={len(train_split)})")

    stratify_col = "emotion" if "emotion" in getattr(train_split, "column_names", []) else None
    kwargs = dict(test_size=val_fraction, seed=seed, shuffle=True)

    # Keep it simple: try stratification if supported; otherwise do a plain random split.
    if stratify_col is not None:
        try:
            split = train_split.train_test_split(**(kwargs | {"stratify_by_column": stratify_col}))
            return split["train"], split["test"]
        except (TypeError, ValueError):
            pass

    split = train_split.train_test_split(**kwargs)
    return split["train"], split["test"]


def _ensure_label_columns(split):
    """Ensure split has emotion/gender/age_category columns for mixed datasets."""
    needed = ("emotion", "gender", "age_category")
    missing = [c for c in needed if c not in split.column_names]
    if missing:
        def _inject_missing(row):
            for c in missing:
                row[c] = None
            return row
        split = split.map(_inject_missing, desc=f"Injecting missing columns: {','.join(missing)}")
    return split


def _prepare_split_for_training(split):
    """Cast audio column and align label columns for downstream dataset."""
    if "audio" in split.column_names:
        split = split.cast_column("audio", Audio(decode=False))
    return _ensure_label_columns(split)


def _force_canonical_label_schema(split, reference_split=None):
    """Normalize columns and cast labels; optionally align exactly to reference features."""
    keep_cols = ["audio", "emotion", "gender", "age_category"]
    split = split.select_columns(keep_cols)
    if reference_split is not None:
        # Match reference dtypes exactly (e.g., large_string vs string/null).
        return split.cast(reference_split.features)

    audio_feature = split.features["audio"] if "audio" in split.features else Audio(decode=False)
    target_features = Features({
        "audio": audio_feature,
        "emotion": Value("string"),
        "gender": Value("string"),
        "age_category": Value("string"),
    })
    return split.cast(target_features)


def _count_label_presence(split):
    """Count non-empty labels per task in a split."""
    counts = {"emotion": 0, "gender": 0, "age": 0}

    def _present(v):
        return v is not None and str(v).strip() != ""

    split_no_audio = split.remove_columns(['audio']) if 'audio' in split.column_names else split
    for row in split_no_audio:
        if _present(row.get("emotion")):
            counts["emotion"] += 1
        if _present(row.get("gender")):
            counts["gender"] += 1
        if _present(row.get("age_category")):
            counts["age"] += 1
    return counts


def _count_emotion_distribution(split):
    counts = {}
    split_no_audio = split.remove_columns(['audio']) if 'audio' in split.column_names else split
    for row in split_no_audio:
        emo = row.get("emotion")
        if emo is None:
            continue
        emo = str(emo).strip()
        if not emo:
            continue
        counts[emo] = counts.get(emo, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def build_mixed_train_val_splits():
    """Build train/val splits using full HF data and optional capped Kazemo data."""
    hf_dataset = load()
    hf_train = hf_dataset.get("train")
    if hf_train is None:
        hf_train = hf_dataset[list(hf_dataset.keys())[0]]

    hf_val = hf_dataset.get('validation', hf_dataset.get('val', hf_dataset.get('test', None)))
    if hf_val is None:
        print(f"Warning: No validation split found on HF. Splitting train into "
              f"train/val with val_fraction={VAL_FRACTION}.")
        hf_train, hf_val = fallback_split_train_val(hf_train, seed=RANDOM_SEED, val_fraction=VAL_FRACTION)

    hf_train = _force_canonical_label_schema(_prepare_split_for_training(hf_train))
    hf_val = _force_canonical_label_schema(_prepare_split_for_training(hf_val))

    train_split = hf_train
    val_split = hf_val
    kazemo_train_count = 0
    kazemo_val_count = 0
    kazemo_emotion_counts = {}

    if USE_KAZEMO:
        print(f"Loading Kazemo dataset (max_samples={KAZEMO_MAX_SAMPLES})...")
        kz_ds: DatasetDict = load_kazemotts(cache_dir=str(DATA_DIR), max_samples=KAZEMO_MAX_SAMPLES)
        kz_base = kz_ds.get("train", kz_ds[list(kz_ds.keys())[0]])
        kz_base = _prepare_split_for_training(kz_base)
        kz_split = kz_base.train_test_split(
            test_size=KAZEMO_VAL_FRACTION,
            seed=RANDOM_SEED,
            shuffle=True,
        )
        kz_train = _force_canonical_label_schema(_prepare_split_for_training(kz_split["train"]), reference_split=hf_train)
        kz_val = _force_canonical_label_schema(_prepare_split_for_training(kz_split["test"]), reference_split=hf_val)
        kazemo_train_count = len(kz_train)
        kazemo_val_count = len(kz_val)
        kazemo_emotion_counts = _count_emotion_distribution(kz_base)

        train_split = concatenate_datasets([hf_train, kz_train])
        val_split = concatenate_datasets([hf_val, kz_val])

    composition = {
        "hf_train": len(hf_train),
        "hf_val": len(hf_val),
        "kazemo_train": kazemo_train_count,
        "kazemo_val": kazemo_val_count,
        "kazemo_emotion_counts": kazemo_emotion_counts,
        "train_total": len(train_split),
        "val_total": len(val_split),
        "train_label_counts": _count_label_presence(train_split),
        "val_label_counts": _count_label_presence(val_split),
    }
    return hf_dataset, train_split, val_split, composition


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


def _spec_augment(
    all_layers: torch.Tensor,
    time_mask_param: int = 70,
    freq_mask_param: int = 27,
    num_time_masks: int = 2,
    num_freq_masks: int = 2,
) -> torch.Tensor:
    """SpecAugment-style masking on stacked hidden states [num_layers, B, T, H].
    LB policy (Park et al. 2019): F=27, T=70, multiplicity 2. Applied online during
    training so the model never sees the same corrupted version twice."""
    _, _, T, H = all_layers.shape
    out = all_layers.clone()
    device = all_layers.device
    # Time masking: up to `time_mask_param` consecutive steps (paper: 40–70)
    for _ in range(num_time_masks):
        t_len = min(time_mask_param, max(1, T - 1))
        t_start = torch.randint(0, max(1, T - t_len + 1), (1,), device=device).item()
        out[:, :, t_start : t_start + t_len, :] = 0.0
    # Frequency masking: up to `freq_mask_param` consecutive channels (paper: 27)
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
    scale_group_decay: bool = False,
    group_power_range: tuple[float, float] = (0.8, 1.2),
) -> torch.optim.lr_scheduler.LambdaLR:
    """Hold LR for hold_epochs, then cosine-anneal to eta_min_factor * base_lr.

    By default, applies the same multiplier to every param group, preserving
    relative LR ratios.

    If scale_group_decay=True, each param group gets its own cosine exponent
    based on its *initial LR*:
    - higher initial LR  -> larger exponent -> faster decay early
    - lower initial LR   -> smaller exponent -> slower decay early

    All groups still reach eta_min_factor * base_lr at the end.
    """
    def base_cosine(epoch: int) -> float:
        if epoch < hold_epochs:
            return 1.0
        # Use (decay_epochs - 1) so the minimum is reached at the very last
        # training epoch rather than one step after it.
        t = min((epoch - hold_epochs) / max(decay_epochs - 1, 1), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    if not scale_group_decay:
        def lr_lambda(epoch: int) -> float:
            c = base_cosine(epoch)
            return eta_min_factor + (1.0 - eta_min_factor) * c
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Per-group exponents based on initial LR rank
    lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]
    lo, hi = min(lrs), max(lrs)
    p_lo, p_hi = group_power_range
    if hi <= 0 or abs(hi - lo) < 1e-12:
        powers = [1.0 for _ in lrs]
    else:
        # normalise to [0,1] by LR; higher LR -> closer to 1
        powers = []
        for lr in lrs:
            r = (lr - lo) / (hi - lo)
            powers.append(p_lo + (p_hi - p_lo) * r)

    lambdas = []
    for p in powers:
        def _f(epoch: int, _p=p) -> float:
            c = base_cosine(epoch)
            return eta_min_factor + (1.0 - eta_min_factor) * (c ** _p)
        lambdas.append(_f)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)


class MultiTaskHubert(nn.Module):
    """HuBERT model with three heads using task-specific weighted layer sums."""

    def __init__(self, num_emotions, num_genders, num_ages, freeze_backbone=True, use_spec_augment=False):
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=True,)
        # Prefer the model's built-in SpecAugment (masks projected features pre-transformer)
        # over manually masking post-transformer hidden states.
        self.use_spec_augment = use_spec_augment
        if hasattr(self.hubert, "config"):
            self.hubert.config.apply_spec_augment = bool(use_spec_augment)
            # Reasonable defaults (HF wav2vec2/HubERT-style)
            self.hubert.config.mask_time_prob = 0.05
            self.hubert.config.mask_time_length = 10
            self.hubert.config.mask_feature_prob = 0.0
            self.hubert.config.mask_feature_length = 10
        if hasattr(self.hubert.config, 'training_drop_path'):
            self.hubert.config.training_drop_path = 0.1

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
            nn.Dropout(0.2),
            nn.Linear(256, num_emotions),
        )
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_genders),
        )
        self.age_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_ages),
        )

    def forward(self, input_values, attention_mask=None):
        """
        Args:
            input_values: Audio features [batch_size, sequence_length]
        Returns:
            emotion_logits, gender_logits, age_logits
        """
        outputs = self.hubert(input_values, attention_mask=attention_mask)

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
                 max_length=160000, is_train: bool = False, noise_dir: str | None = None):  # 10 seconds at 16kHz
        self.data = dataset_split
        self.processor = processor
        self.emotion_encoder = emotion_encoder
        self.gender_encoder = gender_encoder
        self.age_encoder = age_encoder
        self.max_length = max_length
        self.is_train = is_train
        self.speed_factors = (0.9, 1.0, 1.1)
        self.speed_prob = 0.8
        self.noise_prob = 0.5
        self.snr_db_range = (5.0, 20.0)
        self._noise_mixer = None
        if noise_dir:
            from audio_augment import NoiseMixer

            self._noise_mixer = NoiseMixer.from_dir(noise_dir)
    
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

        audio_data = audio_data.astype(np.float32, copy=False)

        # ── Simple training-time waveform augmentations ───────────────────────
        if self.is_train:
            import random
            from audio_augment import speed_perturb, mix_at_snr

            # Speed perturbation (cheap, effective)
            if random.random() < self.speed_prob:
                factor = random.choice(self.speed_factors)
                if factor != 1.0:
                    audio_data = speed_perturb(audio_data, factor)

            # Noise injection from a directory of environmental sounds (e.g. ESC-50)
            if self._noise_mixer is not None and random.random() < self.noise_prob:
                noise = self._noise_mixer.sample(len(audio_data))
                if noise is not None:
                    snr_db = random.uniform(*self.snr_db_range)
                    audio_data = mix_at_snr(audio_data, noise, snr_db)
        
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
            padding=True,
            return_attention_mask=True,
        )
        input_values = inputs.input_values.squeeze(0)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)
        else:
            # Default collate can't batch None; if a mask isn't returned, treat all
            # samples as fully valid (we already hard-pad/truncate to max_length).
            attention_mask = torch.ones(input_values.shape, dtype=torch.long)
        
        # Get labels
        def norm(v):
            return str(v).strip() if (v is not None and str(v).strip()) else None

        emotion_normalized = norm(row.get("emotion"))
        gender_normalized = norm(row.get("gender"))
        age_normalized = norm(row.get("age_category"))

        has_emotion = emotion_normalized is not None
        has_gender = gender_normalized is not None
        has_age = age_normalized is not None

        # Sentinel -100 is ignored by masked training loops for missing labels.
        emotion_label = self.emotion_encoder.get(emotion_normalized, -100) if has_emotion else -100
        gender_label = self.gender_encoder.get(gender_normalized, -100) if has_gender else -100
        age_label = self.age_encoder.get(age_normalized, -100) if has_age else -100

        return {
            'input_values': input_values,
            'attention_mask': attention_mask,
            'emotion': torch.tensor(emotion_label, dtype=torch.long),
            'gender': torch.tensor(gender_label, dtype=torch.long),
            'age': torch.tensor(age_label, dtype=torch.long),
            'has_emotion': torch.tensor(has_emotion, dtype=torch.bool),
            'has_gender': torch.tensor(has_gender, dtype=torch.bool),
            'has_age': torch.tensor(has_age, dtype=torch.bool),
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
    emotion_loss_batches = 0
    gender_loss_batches = 0
    age_loss_batches = 0
    num_batches = 0

    # BF16 autocast — only when the GPU actually supports it
    _use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if _use_bf16 else torch.float32

    for batch in tqdm(dataloader, desc="Training"):
        if _stop_requested:
            break
        # non_blocking=True overlaps H→D transfer with GPU work
        input_values = batch['input_values'].to(device, non_blocking=True)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
        emotion_labels = batch['emotion'].to(device, non_blocking=True)
        gender_labels = batch['gender'].to(device, non_blocking=True)
        age_labels = batch['age'].to(device, non_blocking=True)
        has_emotion = batch['has_emotion'].to(device, non_blocking=True)
        has_gender = batch['has_gender'].to(device, non_blocking=True)
        has_age = batch['has_age'].to(device, non_blocking=True)

        # set_to_none avoids a memset; faster than zeroing
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            emotion_logits, gender_logits, age_logits = model(input_values, attention_mask=attention_mask)

            per_task_losses = {}
            active_weight = 0.0
            total_batch_loss = torch.zeros((), device=device)

            if has_emotion.any():
                loss_emotion = criterion_emotion(emotion_logits[has_emotion], emotion_labels[has_emotion])
                per_task_losses['emotion'] = loss_emotion
                total_batch_loss = total_batch_loss + EMOTION_WEIGHT * loss_emotion
                active_weight += EMOTION_WEIGHT
            if has_gender.any():
                loss_gender = criterion_gender(gender_logits[has_gender], gender_labels[has_gender])
                per_task_losses['gender'] = loss_gender
                total_batch_loss = total_batch_loss + GENDER_WEIGHT * loss_gender
                active_weight += GENDER_WEIGHT
            if has_age.any():
                loss_age = criterion_age(age_logits[has_age], age_labels[has_age])
                per_task_losses['age'] = loss_age
                total_batch_loss = total_batch_loss + AGE_WEIGHT * loss_age
                active_weight += AGE_WEIGHT

            if active_weight > 0.0:
                total_batch_loss = total_batch_loss / active_weight
            else:
                continue

        total_batch_loss.backward()
        # Clip gradients before step (guards against spikes in unfrozen layers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += total_batch_loss.item()
        emotion_loss_sum += float(per_task_losses['emotion'].item()) if 'emotion' in per_task_losses else 0.0
        gender_loss_sum += float(per_task_losses['gender'].item()) if 'gender' in per_task_losses else 0.0
        age_loss_sum += float(per_task_losses['age'].item()) if 'age' in per_task_losses else 0.0
        emotion_loss_batches += 1 if 'emotion' in per_task_losses else 0
        gender_loss_batches += 1 if 'gender' in per_task_losses else 0
        age_loss_batches += 1 if 'age' in per_task_losses else 0
        num_batches += 1

    if num_batches == 0:
        num_batches = 1
    if emotion_loss_batches == 0:
        emotion_loss_batches = 1
    if gender_loss_batches == 0:
        gender_loss_batches = 1
    if age_loss_batches == 0:
        age_loss_batches = 1
    return {
        'total': total_loss / num_batches,
        'emotion': emotion_loss_sum / emotion_loss_batches,
        'gender': gender_loss_sum / gender_loss_batches,
        'age': age_loss_sum / age_loss_batches
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
    emotion_loss_batches = 0
    gender_loss_batches = 0
    age_loss_batches = 0
    
    emotion_correct = 0
    gender_correct = 0
    age_correct = 0
    emotion_samples = 0
    gender_samples = 0
    age_samples = 0
    num_batches = 0

    _use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if _use_bf16 else torch.float32
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if _stop_requested:
                break
            input_values = batch['input_values'].to(device, non_blocking=True)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            emotion_labels = batch['emotion'].to(device, non_blocking=True)
            gender_labels = batch['gender'].to(device, non_blocking=True)
            age_labels = batch['age'].to(device, non_blocking=True)
            has_emotion = batch['has_emotion'].to(device, non_blocking=True)
            has_gender = batch['has_gender'].to(device, non_blocking=True)
            has_age = batch['has_age'].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                emotion_logits, gender_logits, age_logits = model(input_values, attention_mask=attention_mask)

                per_task_losses = {}
                active_weight = 0.0
                total_batch_loss = torch.zeros((), device=device)

                if has_emotion.any():
                    loss_emotion = criterion_emotion(emotion_logits[has_emotion], emotion_labels[has_emotion])
                    per_task_losses['emotion'] = loss_emotion
                    total_batch_loss = total_batch_loss + EMOTION_WEIGHT * loss_emotion
                    active_weight += EMOTION_WEIGHT
                if has_gender.any():
                    loss_gender = criterion_gender(gender_logits[has_gender], gender_labels[has_gender])
                    per_task_losses['gender'] = loss_gender
                    total_batch_loss = total_batch_loss + GENDER_WEIGHT * loss_gender
                    active_weight += GENDER_WEIGHT
                if has_age.any():
                    loss_age = criterion_age(age_logits[has_age], age_labels[has_age])
                    per_task_losses['age'] = loss_age
                    total_batch_loss = total_batch_loss + AGE_WEIGHT * loss_age
                    active_weight += AGE_WEIGHT

                if active_weight > 0.0:
                    total_batch_loss = total_batch_loss / active_weight
                else:
                    continue

            total_loss += total_batch_loss.item()
            emotion_loss_sum += float(per_task_losses['emotion'].item()) if 'emotion' in per_task_losses else 0.0
            gender_loss_sum += float(per_task_losses['gender'].item()) if 'gender' in per_task_losses else 0.0
            age_loss_sum += float(per_task_losses['age'].item()) if 'age' in per_task_losses else 0.0
            emotion_loss_batches += 1 if 'emotion' in per_task_losses else 0
            gender_loss_batches += 1 if 'gender' in per_task_losses else 0
            age_loss_batches += 1 if 'age' in per_task_losses else 0

            # Compute accuracy
            if has_emotion.any():
                emotion_pred = torch.argmax(emotion_logits[has_emotion], dim=1)
                emotion_correct += (emotion_pred == emotion_labels[has_emotion]).sum().item()
                emotion_samples += int(has_emotion.sum().item())
            if has_gender.any():
                gender_pred = torch.argmax(gender_logits[has_gender], dim=1)
                gender_correct += (gender_pred == gender_labels[has_gender]).sum().item()
                gender_samples += int(has_gender.sum().item())
            if has_age.any():
                age_pred = torch.argmax(age_logits[has_age], dim=1)
                age_correct += (age_pred == age_labels[has_age]).sum().item()
                age_samples += int(has_age.sum().item())
            num_batches += 1

    if num_batches == 0:
        num_batches = 1
    if emotion_loss_batches == 0:
        emotion_loss_batches = 1
    if gender_loss_batches == 0:
        gender_loss_batches = 1
    if age_loss_batches == 0:
        age_loss_batches = 1
    if emotion_samples == 0:
        emotion_samples = 1
    if gender_samples == 0:
        gender_samples = 1
    if age_samples == 0:
        age_samples = 1
    return {
        'total': total_loss / num_batches,
        'emotion': emotion_loss_sum / emotion_loss_batches,
        'gender': gender_loss_sum / gender_loss_batches,
        'age': age_loss_sum / age_loss_batches,
        'emotion_acc': emotion_correct / emotion_samples,
        'gender_acc': gender_correct / gender_samples,
        'age_acc': age_correct / age_samples
    }


def main():
    global _stop_requested
    set_seed(RANDOM_SEED)
    signal.signal(signal.SIGINT, _sigint_handler)
    print(f"Using device: {DEVICE}")
    print("Press Ctrl+C to stop training and save a checkpoint.")

    # Load dataset and construct mixed HF+Kazemo train/val splits
    dataset, train_split, val_split, composition = build_mixed_train_val_splits()
    merged_for_encoders = DatasetDict({"train": train_split, "validation": val_split})
    
    # Build label encoders
    print("Building label encoders...")
    emotion_encoder, gender_encoder, age_encoder = build_label_encoders(merged_for_encoders)
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
    
    print("Dataset composition:")
    print(f"  HF train/val: {composition['hf_train']} / {composition['hf_val']}")
    print(f"  Kazemo train/val (cap={KAZEMO_MAX_SAMPLES}, enabled={USE_KAZEMO}): "
          f"{composition['kazemo_train']} / {composition['kazemo_val']}")
    if composition.get("kazemo_emotion_counts"):
        emo_counts = ", ".join([f"{k}:{v}" for k, v in composition["kazemo_emotion_counts"].items()])
        print(f"  Kazemo selected emotion counts: {emo_counts}")
    print(f"  Mixed train/val total: {composition['train_total']} / {composition['val_total']}")
    print(f"  Train labels present: emotion={composition['train_label_counts']['emotion']}, "
          f"gender={composition['train_label_counts']['gender']}, "
          f"age={composition['train_label_counts']['age']}")
    print(f"  Val labels present: emotion={composition['val_label_counts']['emotion']}, "
          f"gender={composition['val_label_counts']['gender']}, "
          f"age={composition['val_label_counts']['age']}")
    
    train_dataset = AudioDataset(
        train_split,
        processor,
        emotion_encoder,
        gender_encoder,
        age_encoder,
        is_train=True,
        noise_dir=os.environ.get("NOISE_DIR"),
    )

    val_dataset = AudioDataset(
        val_split,
        processor,
        emotion_encoder,
        gender_encoder,
        age_encoder,
        is_train=False,
        noise_dir=None,
    )
    
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

    # Loss functions: label smoothing to reduce overfitting to noisy labels and rising val loss
    criterion_emotion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_gender = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_age = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Resume from latest checkpoint if training didn't finish
    metrics_path = MODEL_DIR / "training_metrics.json"
    latest_path = MODEL_DIR / "latest_checkpoint.pt"
    best_val_loss = float('inf')
    last_epoch = -1
    all_metrics = []
    start_epoch = 0
    epochs_without_improvement = 0

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

        # Save best model and early stopping
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            epochs_without_improvement = 0
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
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping: no val loss improvement for {EARLY_STOPPING_PATIENCE} epochs.")
                break

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
