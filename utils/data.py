"""HF dataset prep helpers, PyTorch AudioDataset, and label encoder builder."""

from __future__ import annotations

import numpy as np
import torch
from datasets import Audio, Dataset, Features, Value, concatenate_datasets
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from loaders.load_data import read_audio
from utils.misc import SAMPLE_RATE, RANDOM_SEED

VAL_FRACTION = 0.1


# ── HF dataset preparation ───────────────────────────────────────────────────

def fallback_split_train_val(
    train_split, *, seed: int = RANDOM_SEED, val_fraction: float = VAL_FRACTION
):
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
    counts: dict[str, int] = {}
    split_no_audio = split.remove_columns(["audio"]) if "audio" in split.column_names else split
    for row in split_no_audio:
        emo = row.get("emotion")
        if emo is None:
            continue
        emo = str(emo).strip()
        if not emo:
            continue
        counts[emo] = counts.get(emo, 0) + 1
    return dict(sorted(counts.items()))


# ── PyTorch dataset ──────────────────────────────────────────────────────────

class AudioDataset(TorchDataset):
    """PyTorch Dataset for multi-task audio classification (emotion / gender / age)."""

    def __init__(
        self,
        dataset_split,
        processor,
        emotion_encoder,
        gender_encoder,
        age_encoder,
        max_length: int = 160000,
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

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "emotion": torch.tensor(
                self.emotion_encoder.get(emotion_normalized, -100) if has_emotion else -100,
                dtype=torch.long,
            ),
            "gender": torch.tensor(
                self.gender_encoder.get(gender_normalized, -100) if has_gender else -100,
                dtype=torch.long,
            ),
            "age": torch.tensor(
                self.age_encoder.get(age_normalized, -100) if has_age else -100,
                dtype=torch.long,
            ),
            "has_emotion": torch.tensor(has_emotion, dtype=torch.bool),
            "has_gender": torch.tensor(has_gender, dtype=torch.bool),
            "has_age": torch.tensor(has_age, dtype=torch.bool),
        }


def build_label_encoders(dataset):
    emotions: set[str] = set()
    genders: set[str] = set()
    age_categories: set[str] = set()

    for split_name in dataset.keys():
        split = dataset[split_name]
        split_no_audio = split.remove_columns(["audio"]) if "audio" in split.column_names else split
        for row in split_no_audio:
            if row.get("emotion") and str(row["emotion"]).strip():
                emotions.add(str(row["emotion"]).strip())
            if row.get("gender") and str(row["gender"]).strip():
                genders.add(str(row["gender"]).strip())
            if row.get("age_category") and str(row["age_category"]).strip():
                age_categories.add(str(row["age_category"]).strip())

    emotion_encoder = {lbl: idx for idx, lbl in enumerate(sorted(emotions))}
    gender_encoder = {lbl: idx for idx, lbl in enumerate(sorted(genders))}
    age_encoder = {lbl: idx for idx, lbl in enumerate(sorted(age_categories))}
    return emotion_encoder, gender_encoder, age_encoder
