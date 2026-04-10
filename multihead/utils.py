"""Shared data-loading, training-loop, and checkpoint helpers for multi-task audio training."""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Audio, DatasetDict, Features, Value, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login as hf_login
from torch.utils.data import Dataset
from tqdm import tqdm

from eval.validation_holdout import (
    first_index_by_row_id,
    holdout_source_id_set,
    load_validation_sample_ids,
    train_indices_excluding_holdout,
    val_indices_from_manifest,
)
from loaders.kazemo.load_data import load_kazemotts
from loaders.load_data import DATA_DIR, load, read_audio

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

# ── Paths & sample rate ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16_000  # HuBERT requires 16kHz

RANDOM_SEED = 42
VAL_FRACTION = 0.1
KAZEMO_MAX_SAMPLES = int(os.environ.get("KAZEMO_MAX_SAMPLES", "20000"))
KAZEMO_VAL_FRACTION = float(os.environ.get("KAZEMO_VAL_FRACTION", "0.1"))
USE_KAZEMO = os.environ.get("USE_KAZEMO", "1").strip().lower() not in {"0", "false", "no"}

TRAIN_VAL_MANIFEST = os.environ.get("TRAIN_VAL_MANIFEST", "").strip()
HF_BATCH01_ID = os.environ.get("HF_BATCH01_ID", "01gumano1d/batch01-validation-test")
HF_BATCH02_ID = os.environ.get("HF_BATCH02_ID", "01gumano1d/batch2-aug-clean")
HF_BATCH01_SPLIT = os.environ.get("HF_BATCH01_SPLIT", "train")
HF_BATCH01_CACHE = Path(
    os.environ.get("HF_BATCH01_CACHE", str(REPO_ROOT / "data" / "batch01-validation-test"))
)
HF_BATCH02_CACHE = Path(
    os.environ.get("HF_BATCH02_CACHE", str(REPO_ROOT / "data" / "batch2-aug-clean"))
)

# Safe stop: set by SIGINT handler (Ctrl+C)
stop_requested = False


def apply_cuda_perf_flags(device: torch.device | None = None) -> None:
    """TF32 / cuDNN / matmul precision defaults for CUDA training."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set global seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sigint_handler(signum, frame) -> None:
    global stop_requested
    if stop_requested:
        print("\nSecond Ctrl+C: exiting immediately.", file=sys.stderr)
        sys.exit(130)
    stop_requested = True
    print("\nCtrl+C received. Finishing current batch and saving checkpoint...")


# Backward compatibility: legacy private name
_sigint_handler = sigint_handler


def unwrap(model: nn.Module) -> nn.Module:
    """Return the original module, unwrapping torch.compile's OptimizedModule if present."""
    return getattr(model, "_orig_mod", model)


# Backward compatibility
_unwrap = unwrap


def save_wandb_file_artifact(run, *, file_path: Path, name: str, artifact_type: str) -> None:
    if run is None or wandb is None:
        return
    if not file_path.exists():
        return
    art = wandb.Artifact(name=name, type=artifact_type)
    art.add_file(str(file_path), name=file_path.name)
    run.log_artifact(art)


def save_step_checkpoint(
    *,
    step_dir: Path,
    global_step: int,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    best_val_loss: float,
    num_emotions: int,
    num_genders: int,
    num_ages: int,
    samples_seen: int,
) -> Path:
    step_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = step_dir / f"checkpoint_step_{global_step:08d}.pt"
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "samples_seen": samples_seen,
        "source": "step",
        "model_state_dict": unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "num_emotions": num_emotions,
        "num_genders": num_genders,
        "num_ages": num_ages,
    }, ckpt_path)
    return ckpt_path


def rotate_step_checkpoints(step_dir: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    files = sorted(step_dir.glob("checkpoint_step_*.pt"))
    stale = files[:-keep_last_n]
    for p in stale:
        try:
            p.unlink()
        except OSError:
            pass


_save_wandb_file_artifact = save_wandb_file_artifact
_save_step_checkpoint = save_step_checkpoint
_rotate_step_checkpoints = rotate_step_checkpoints


def make_cosine_schedule(
    optimizer: optim.Optimizer,
    hold_epochs: int,
    decay_epochs: int,
    eta_min_factor: float = 1e-2,
    scale_group_decay: bool = False,
    group_power_range: tuple[float, float] = (0.8, 1.2),
) -> torch.optim.lr_scheduler.LambdaLR:
    """Hold LR for hold_epochs, then cosine-anneal to eta_min_factor * base_lr."""

    def base_cosine(epoch: int) -> float:
        if epoch < hold_epochs:
            return 1.0
        t = min((epoch - hold_epochs) / max(decay_epochs - 1, 1), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    if not scale_group_decay:
        def lr_lambda(epoch: int) -> float:
            c = base_cosine(epoch)
            return eta_min_factor + (1.0 - eta_min_factor) * c
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]
    lo, hi = min(lrs), max(lrs)
    p_lo, p_hi = group_power_range
    if hi <= 0 or abs(hi - lo) < 1e-12:
        powers = [1.0 for _ in lrs]
    else:
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


_make_cosine_schedule = make_cosine_schedule


def fallback_split_train_val(train_split, *, seed: int = RANDOM_SEED, val_fraction: float = VAL_FRACTION):
    """Create a disjoint (train, val) split from a single HF Dataset."""
    if train_split is None:
        raise ValueError("train_split must not be None")
    if len(train_split) < 2:
        raise ValueError(f"Not enough rows to split train/val (n={len(train_split)})")

    stratify_col = "emotion" if "emotion" in getattr(train_split, "column_names", []) else None
    kwargs = dict(test_size=val_fraction, seed=seed, shuffle=True)

    if stratify_col is not None:
        try:
            split = train_split.train_test_split(**(kwargs | {"stratify_by_column": stratify_col}))
            return split["train"], split["test"]
        except (TypeError, ValueError):
            pass

    split = train_split.train_test_split(**kwargs)
    return split["train"], split["test"]


def _ensure_label_columns(split):
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
    if "audio" in split.column_names:
        split = split.cast_column("audio", Audio(decode=False))
    return _ensure_label_columns(split)


def _force_canonical_label_schema(split, reference_split=None):
    keep_cols = ["audio", "emotion", "gender", "age_category"]
    split = split.select_columns(keep_cols)
    if reference_split is not None:
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
    counts = {"emotion": 0, "gender": 0, "age": 0}

    def _present(v):
        return v is not None and str(v).strip() != ""

    split_no_audio = split.remove_columns(["audio"]) if "audio" in split.column_names else split
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
    split_no_audio = split.remove_columns(["audio"]) if "audio" in split.column_names else split
    for row in split_no_audio:
        emo = row.get("emotion")
        if emo is None:
            continue
        emo = str(emo).strip()
        if not emo:
            continue
        counts[emo] = counts.get(emo, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def build_holdout_mixed_train_val_splits(manifest_path: Path):
    load_dotenv(REPO_ROOT / ".env")
    tok = os.environ.get("HF_TOKEN")
    if tok:
        hf_login(token=tok)

    manifest_path = manifest_path.resolve()
    sample_ids = load_validation_sample_ids(manifest_path)
    holdout_bases = holdout_source_id_set(sample_ids)

    print(f"[data] Holdout manifest: {manifest_path} ({len(sample_ids)} validation rows)")
    HF_BATCH01_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"[data] batch01: {HF_BATCH01_ID!r} cache={HF_BATCH01_CACHE} split={HF_BATCH01_SPLIT!r}")
    ds1 = load_dataset(HF_BATCH01_ID, cache_dir=str(HF_BATCH01_CACHE))
    split1 = ds1[HF_BATCH01_SPLIT] if HF_BATCH01_SPLIT in ds1 else ds1[list(ds1.keys())[0]]
    split1 = _force_canonical_label_schema(_prepare_split_for_training(split1))

    id_to_idx = first_index_by_row_id(split1)
    val_indices = val_indices_from_manifest(sample_ids, id_to_idx)
    val_split = split1.select(val_indices)
    if len(val_split) != len(sample_ids):
        raise RuntimeError(f"Val size {len(val_split)} != manifest {len(sample_ids)}")

    train_idx = train_indices_excluding_holdout(split1, holdout_bases)
    train_b1 = split1.select(train_idx)

    HF_BATCH02_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"[data] batch02 → train only: {HF_BATCH02_ID!r} cache={HF_BATCH02_CACHE}")
    ds2 = load_dataset(HF_BATCH02_ID, cache_dir=str(HF_BATCH02_CACHE))
    b2 = ds2.get("train")
    if b2 is None:
        b2 = ds2[list(ds2.keys())[0]]
    b2 = _force_canonical_label_schema(_prepare_split_for_training(b2), reference_split=train_b1)

    kazemo_train_count = 0
    kazemo_val_count = 0
    kazemo_emotion_counts: dict = {}

    holdout_val = val_split  # batch01 holdout only — saved before any Kazemo concat
    train_split = concatenate_datasets([train_b1, b2])
    named_val_splits: dict = {}

    if USE_KAZEMO:
        print(f"[data] USE_KAZEMO=1: appending Kazemo to train+val (fraction={KAZEMO_VAL_FRACTION}); cap={KAZEMO_MAX_SAMPLES}")
        kz_ds: DatasetDict = load_kazemotts(cache_dir=str(DATA_DIR), max_samples=KAZEMO_MAX_SAMPLES)
        kz_base = kz_ds.get("train", kz_ds[list(kz_ds.keys())[0]])
        kz_base = _prepare_split_for_training(kz_base)
        kz_split = kz_base.train_test_split(
            test_size=KAZEMO_VAL_FRACTION,
            seed=RANDOM_SEED,
            shuffle=True,
        )
        kz_train = _force_canonical_label_schema(
            _prepare_split_for_training(kz_split["train"]), reference_split=train_b1
        )
        kz_val = _force_canonical_label_schema(
            _prepare_split_for_training(kz_split["test"]), reference_split=holdout_val
        )
        kazemo_train_count = len(kz_train)
        kazemo_val_count = len(kz_val)
        kazemo_emotion_counts = _count_emotion_distribution(kz_base)
        train_split = concatenate_datasets([train_split, kz_train])
        val_split = concatenate_datasets([holdout_val, kz_val])
        named_val_splits = {"val": holdout_val, "kazemo": kz_val}
    else:
        named_val_splits = {"val": holdout_val}

    composition = {
        "mode": "holdout_manifest",
        "manifest": str(manifest_path),
        "batch01_id": HF_BATCH01_ID,
        "batch02_id": HF_BATCH02_ID,
        "batch01_split": HF_BATCH01_SPLIT,
        "batch01_train_only": len(train_b1),
        "batch02_train_only": len(b2),
        "hf_train": len(train_b1),
        "hf_val": len(holdout_val),
        "kazemo_train": kazemo_train_count,
        "kazemo_val": kazemo_val_count,
        "kazemo_emotion_counts": kazemo_emotion_counts,
        "train_total": len(train_split),
        "val_total": len(val_split),
        "train_label_counts": _count_label_presence(train_split),
        "val_label_counts": _count_label_presence(val_split),
    }
    merged_hf = DatasetDict({"batch01_train": train_b1, "batch01_val_holdout": holdout_val, "batch02_train": b2})
    return merged_hf, train_split, val_split, named_val_splits, composition


def _resolve_train_val_manifest_path(raw: str) -> Path:
    """Resolve TRAIN_VAL_MANIFEST: prefer configured path, then ``results/``, then legacy ``eval/``."""
    mp = Path(raw)
    if not mp.is_absolute():
        mp = REPO_ROOT / mp
    resolved = mp.resolve()
    if resolved.exists():
        return resolved
    for alt in (REPO_ROOT / "results" / mp.name, REPO_ROOT / "eval" / mp.name):
        if alt.exists():
            print(
                f"[data] Holdout manifest: using {alt} (configured path missing: {resolved})",
                flush=True,
            )
            return alt.resolve()
    raise FileNotFoundError(
        f"Holdout manifest not found. Tried {resolved}, "
        f"{REPO_ROOT / 'results' / mp.name}, {REPO_ROOT / 'eval' / mp.name}. "
        "Run eval/validate.py (writes under results/) or set TRAIN_VAL_MANIFEST."
    )


def build_mixed_train_val_splits():
    if TRAIN_VAL_MANIFEST:
        mp = _resolve_train_val_manifest_path(TRAIN_VAL_MANIFEST)
        return build_holdout_mixed_train_val_splits(mp)

    hf_dataset = load()
    hf_train = hf_dataset.get("train")
    if hf_train is None:
        hf_train = hf_dataset[list(hf_dataset.keys())[0]]

    hf_val = hf_dataset.get("validation", hf_dataset.get("val", hf_dataset.get("test", None)))
    if hf_val is None:
        print(
            f"Warning: No validation split found on HF. Splitting train into "
            f"train/val with val_fraction={VAL_FRACTION}."
        )
        hf_train, hf_val = fallback_split_train_val(hf_train, seed=RANDOM_SEED, val_fraction=VAL_FRACTION)

    hf_train = _force_canonical_label_schema(_prepare_split_for_training(hf_train))
    hf_val = _force_canonical_label_schema(_prepare_split_for_training(hf_val))

    train_split = hf_train
    val_split = hf_val
    kazemo_train_count = 0
    kazemo_val_count = 0
    kazemo_emotion_counts = {}

    named_val_splits: dict = {}

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
        named_val_splits = {"val": hf_val, "kazemo": kz_val}
    else:
        named_val_splits = {"val": val_split}

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
    return hf_dataset, train_split, val_split, named_val_splits, composition


class AudioDataset(Dataset):
    """Dataset for audio classification with emotion, gender, and age_category labels."""

    def __init__(
        self,
        dataset_split,
        processor,
        emotion_encoder,
        gender_encoder,
        age_encoder,
        max_length=160000,
        is_train: bool = False,
        noise_dir: str | None = None,
    ):
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
            from utils.audio_augment import NoiseMixer

            self._noise_mixer = NoiseMixer.from_dir(noise_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        audio_data, sample_rate = read_audio(row["audio"])

        if sample_rate != SAMPLE_RATE:
            import librosa
            audio_data = librosa.resample(
                audio_data.astype(np.float32),
                orig_sr=sample_rate,
                target_sr=SAMPLE_RATE,
            )

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        audio_data = audio_data.astype(np.float32, copy=False)

        if self.is_train:
            import random as _random
            from utils.audio_augment import mix_at_snr, speed_perturb

            if _random.random() < self.speed_prob:
                factor = _random.choice(self.speed_factors)
                if factor != 1.0:
                    audio_data = speed_perturb(audio_data, factor)

            if self._noise_mixer is not None and _random.random() < self.noise_prob:
                noise = self._noise_mixer.sample(len(audio_data))
                if noise is not None:
                    snr_db = _random.uniform(*self.snr_db_range)
                    audio_data = mix_at_snr(audio_data, noise, snr_db)

        if len(audio_data) > self.max_length:
            audio_data = audio_data[: self.max_length]
        else:
            audio_data = np.pad(audio_data, (0, self.max_length - len(audio_data)))

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
            attention_mask = torch.ones(input_values.shape, dtype=torch.long)

        def norm(v):
            return str(v).strip() if (v is not None and str(v).strip()) else None

        emotion_normalized = norm(row.get("emotion"))
        gender_normalized = norm(row.get("gender"))
        age_normalized = norm(row.get("age_category"))

        has_emotion = emotion_normalized is not None
        has_gender = gender_normalized is not None
        has_age = age_normalized is not None

        emotion_label = self.emotion_encoder.get(emotion_normalized, -100) if has_emotion else -100
        gender_label = self.gender_encoder.get(gender_normalized, -100) if has_gender else -100
        age_label = self.age_encoder.get(age_normalized, -100) if has_age else -100

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "emotion": torch.tensor(emotion_label, dtype=torch.long),
            "gender": torch.tensor(gender_label, dtype=torch.long),
            "age": torch.tensor(age_label, dtype=torch.long),
            "has_emotion": torch.tensor(has_emotion, dtype=torch.bool),
            "has_gender": torch.tensor(has_gender, dtype=torch.bool),
            "has_age": torch.tensor(has_age, dtype=torch.bool),
        }


def build_label_encoders(dataset):
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
            if "age_category" in row and row["age_category"] is not None and str(row["age_category"]).strip():
                age_categories.add(str(row["age_category"]).strip())
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


def train_epoch(
    model,
    dataloader,
    criterion_emotion,
    criterion_gender,
    criterion_age,
    optimizer,
    device,
    *,
    step_state=None,
    on_batch_end=None,
    epoch_index: int = -1,
    emotion_weight: float = 1.2,
    gender_weight: float = 0.5,
    age_weight: float = 1.0,
    grad_clip_norm: float = 1.0,
):
    """Train for one epoch. Returns (metrics_dict, was_stopped)."""
    global stop_requested
    model.train()
    total_loss = 0.0
    emotion_loss_sum = 0.0
    gender_loss_sum = 0.0
    age_loss_sum = 0.0
    emotion_loss_batches = 0
    gender_loss_batches = 0
    age_loss_batches = 0
    num_batches = 0

    emotion_correct = 0
    gender_correct = 0
    age_correct = 0
    emotion_samples = 0
    gender_samples = 0
    age_samples = 0

    _use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if _use_bf16 else torch.float32

    for batch in tqdm(dataloader, desc="Training"):
        if stop_requested:
            break
        input_values = batch["input_values"].to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
        emotion_labels = batch["emotion"].to(device, non_blocking=True)
        gender_labels = batch["gender"].to(device, non_blocking=True)
        age_labels = batch["age"].to(device, non_blocking=True)
        has_emotion = batch["has_emotion"].to(device, non_blocking=True)
        has_gender = batch["has_gender"].to(device, non_blocking=True)
        has_age = batch["has_age"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            emotion_logits, gender_logits, age_logits = model(input_values, attention_mask=attention_mask)

            per_task_losses = {}
            active_weight = 0.0
            total_batch_loss = torch.zeros((), device=device)

            if has_emotion.any():
                loss_emotion = criterion_emotion(emotion_logits[has_emotion], emotion_labels[has_emotion])
                per_task_losses["emotion"] = loss_emotion
                total_batch_loss = total_batch_loss + emotion_weight * loss_emotion
                active_weight += emotion_weight
            if has_gender.any():
                loss_gender = criterion_gender(gender_logits[has_gender], gender_labels[has_gender])
                per_task_losses["gender"] = loss_gender
                total_batch_loss = total_batch_loss + gender_weight * loss_gender
                active_weight += gender_weight
            if has_age.any():
                loss_age = criterion_age(age_logits[has_age], age_labels[has_age])
                per_task_losses["age"] = loss_age
                total_batch_loss = total_batch_loss + age_weight * loss_age
                active_weight += age_weight

            if active_weight > 0.0:
                total_batch_loss = total_batch_loss / active_weight
            else:
                continue

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

        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        current_step = None
        if step_state is not None:
            step_state["global_step"] = int(step_state.get("global_step", 0)) + 1
            step_state["samples_seen"] = int(step_state.get("samples_seen", 0)) + int(input_values.shape[0])
            current_step = int(step_state["global_step"])

        if on_batch_end is not None and current_step is not None:
            on_batch_end({
                "global_step": current_step,
                "epoch": epoch_index,
                "train_total_loss": float(total_batch_loss.item()),
                "train_emotion_loss": float(per_task_losses["emotion"].item()) if "emotion" in per_task_losses else None,
                "train_gender_loss": float(per_task_losses["gender"].item()) if "gender" in per_task_losses else None,
                "train_age_loss": float(per_task_losses["age"].item()) if "age" in per_task_losses else None,
            })

        total_loss += total_batch_loss.item()
        emotion_loss_sum += float(per_task_losses["emotion"].item()) if "emotion" in per_task_losses else 0.0
        gender_loss_sum += float(per_task_losses["gender"].item()) if "gender" in per_task_losses else 0.0
        age_loss_sum += float(per_task_losses["age"].item()) if "age" in per_task_losses else 0.0
        emotion_loss_batches += 1 if "emotion" in per_task_losses else 0
        gender_loss_batches += 1 if "gender" in per_task_losses else 0
        age_loss_batches += 1 if "age" in per_task_losses else 0
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
        "total": total_loss / num_batches,
        "emotion": emotion_loss_sum / emotion_loss_batches,
        "gender": gender_loss_sum / gender_loss_batches,
        "age": age_loss_sum / age_loss_batches,
        "emotion_acc": emotion_correct / emotion_samples,
        "gender_acc": gender_correct / gender_samples,
        "age_acc": age_correct / age_samples,
    }, stop_requested


def validate(
    model,
    dataloader,
    criterion_emotion,
    criterion_gender,
    criterion_age,
    device,
    *,
    emotion_weight: float = 1.2,
    gender_weight: float = 0.5,
    age_weight: float = 1.0,
):
    global stop_requested
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

    _use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if _use_bf16 else torch.float32
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if stop_requested:
                break
            input_values = batch["input_values"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            emotion_labels = batch["emotion"].to(device, non_blocking=True)
            gender_labels = batch["gender"].to(device, non_blocking=True)
            age_labels = batch["age"].to(device, non_blocking=True)
            has_emotion = batch["has_emotion"].to(device, non_blocking=True)
            has_gender = batch["has_gender"].to(device, non_blocking=True)
            has_age = batch["has_age"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                emotion_logits, gender_logits, age_logits = model(input_values, attention_mask=attention_mask)

                per_task_losses = {}
                active_weight = 0.0
                total_batch_loss = torch.zeros((), device=device)

                if has_emotion.any():
                    loss_emotion = criterion_emotion(emotion_logits[has_emotion], emotion_labels[has_emotion])
                    per_task_losses["emotion"] = loss_emotion
                    total_batch_loss = total_batch_loss + emotion_weight * loss_emotion
                    active_weight += emotion_weight
                if has_gender.any():
                    loss_gender = criterion_gender(gender_logits[has_gender], gender_labels[has_gender])
                    per_task_losses["gender"] = loss_gender
                    total_batch_loss = total_batch_loss + gender_weight * loss_gender
                    active_weight += gender_weight
                if has_age.any():
                    loss_age = criterion_age(age_logits[has_age], age_labels[has_age])
                    per_task_losses["age"] = loss_age
                    total_batch_loss = total_batch_loss + age_weight * loss_age
                    active_weight += age_weight

                if active_weight > 0.0:
                    total_batch_loss = total_batch_loss / active_weight
                else:
                    continue

            total_loss += total_batch_loss.item()
            emotion_loss_sum += float(per_task_losses["emotion"].item()) if "emotion" in per_task_losses else 0.0
            gender_loss_sum += float(per_task_losses["gender"].item()) if "gender" in per_task_losses else 0.0
            age_loss_sum += float(per_task_losses["age"].item()) if "age" in per_task_losses else 0.0
            emotion_loss_batches += 1 if "emotion" in per_task_losses else 0
            gender_loss_batches += 1 if "gender" in per_task_losses else 0
            age_loss_batches += 1 if "age" in per_task_losses else 0

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
        "total": total_loss / num_batches,
        "emotion": emotion_loss_sum / emotion_loss_batches,
        "gender": gender_loss_sum / gender_loss_batches,
        "age": age_loss_sum / age_loss_batches,
        "emotion_acc": emotion_correct / emotion_samples,
        "gender_acc": gender_correct / gender_samples,
        "age_acc": age_correct / age_samples,
    }


def make_batch_end_handler(
    *,
    # mutable shared state
    step_state: dict,
    train_state: dict,
    all_step_val_metrics: list,
    # model / optimiser / scheduler
    model,
    optimizer,
    scheduler,
    num_emotions: int,
    num_genders: int,
    num_ages: int,
    # checkpointing
    checkpoint_every_steps: int,
    checkpoint_keep_last_n: int,
    checkpoint_save_latest_every_steps: bool,
    step_ckpt_dir: Path,
    latest_path: Path,
    step_val_metrics_path: Path,
    # step-level validation
    val_every_steps: int,
    val_loaders: dict,
    criterion_emotion,
    criterion_gender,
    criterion_age,
    device,
    emotion_weight: float,
    gender_weight: float,
    age_weight: float,
    # wandb
    wandb_run,
    wandb_upload_step_artifact: bool,
    wandb_upload_latest_artifact: bool,
    wandb_latest_artifact_every_steps: int,
    step_artifact_name: str,
    latest_artifact_name: str,
):
    """Return an ``on_batch_end`` callback suitable for passing to ``train_epoch``.

    All configuration is captured at factory-call time.  ``train_state`` and
    ``step_state`` are read at handler-call time so the handler always sees the
    current ``best_val_loss`` and ``global_step``.
    """
    def _handler(payload):
        global_step = payload['global_step']

        # ── W&B per-batch train loss ─────────────────────────────────────────
        if wandb_run is not None:
            data = {
                'train/loss_total': payload['train_total_loss'],
                'train/epoch': payload['epoch'] + 1,
            }
            if payload['train_emotion_loss'] is not None:
                data['train/loss_emotion'] = payload['train_emotion_loss']
            if payload['train_gender_loss'] is not None:
                data['train/loss_gender'] = payload['train_gender_loss']
            if payload['train_age_loss'] is not None:
                data['train/loss_age'] = payload['train_age_loss']
            wandb_run.log(data, step=global_step)

        # ── Step checkpoint ──────────────────────────────────────────────────
        if checkpoint_every_steps > 0 and global_step % checkpoint_every_steps == 0:
            step_path = save_step_checkpoint(
                step_dir=step_ckpt_dir,
                global_step=global_step,
                epoch=payload['epoch'],
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=train_state['best_val_loss'],
                num_emotions=num_emotions,
                num_genders=num_genders,
                num_ages=num_ages,
                samples_seen=step_state['samples_seen'],
            )
            rotate_step_checkpoints(step_ckpt_dir, checkpoint_keep_last_n)
            print(f"Saved step checkpoint: {step_path.name}")

            if checkpoint_save_latest_every_steps:
                torch.save({
                    'epoch': payload['epoch'],
                    'global_step': global_step,
                    'samples_seen': step_state['samples_seen'],
                    'model_state_dict': unwrap(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': train_state['best_val_loss'],
                    'num_emotions': num_emotions,
                    'num_genders': num_genders,
                    'num_ages': num_ages,
                }, latest_path)

            if wandb_run is not None and wandb_upload_step_artifact:
                save_wandb_file_artifact(
                    wandb_run,
                    file_path=step_path,
                    name=step_artifact_name,
                    artifact_type='checkpoint',
                )

            if (wandb_run is not None and wandb_upload_latest_artifact
                    and wandb_latest_artifact_every_steps > 0
                    and global_step % wandb_latest_artifact_every_steps == 0
                    and latest_path.exists()):
                save_wandb_file_artifact(
                    wandb_run,
                    file_path=latest_path,
                    name=latest_artifact_name,
                    artifact_type='checkpoint',
                )

        # ── Step-level validation ────────────────────────────────────────────
        if val_every_steps > 0 and global_step % val_every_steps == 0:
            step_record: dict = {"global_step": global_step, "epoch": payload['epoch'] + 1}
            wandb_data: dict = {}
            for _name, _vloader in val_loaders.items():
                _prefix = "val" if _name == "val" else f"val_{_name}"
                _sv = validate(
                    model, _vloader, criterion_emotion, criterion_gender,
                    criterion_age, device,
                    emotion_weight=emotion_weight,
                    gender_weight=gender_weight,
                    age_weight=age_weight,
                )
                model.train()  # validate() leaves model in eval mode
                print(
                    f"  [Step {global_step}][{_name}] Val Loss: {_sv['total']:.4f}  "
                    f"Emotion Acc: {_sv['emotion_acc']:.4f}  "
                    f"Gender Acc: {_sv['gender_acc']:.4f}  "
                    f"Age Acc: {_sv['age_acc']:.4f}"
                )
                wandb_data.update({
                    f'{_prefix}/loss_total': float(_sv['total']),
                    f'{_prefix}/loss_emotion': float(_sv['emotion']),
                    f'{_prefix}/loss_gender': float(_sv['gender']),
                    f'{_prefix}/loss_age': float(_sv['age']),
                    f'{_prefix}/acc_emotion': float(_sv['emotion_acc']),
                    f'{_prefix}/acc_gender': float(_sv['gender_acc']),
                    f'{_prefix}/acc_age': float(_sv['age_acc']),
                })
                step_record[_prefix] = {k: round(float(v), 6) for k, v in _sv.items()}
            if wandb_run is not None:
                wandb_run.log(wandb_data, step=global_step)
            all_step_val_metrics.append(step_record)
            with open(step_val_metrics_path, "w") as f:
                json.dump(all_step_val_metrics, f, indent=2)

    return _handler
