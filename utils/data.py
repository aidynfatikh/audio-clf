"""HF dataset prep helpers, PyTorch AudioDataset, and label encoder builder."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from loaders.load_data import read_audio
from utils.misc import SAMPLE_RATE


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

        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        audio_data = audio_data.astype(np.float32, copy=False)

        if sample_rate != SAMPLE_RATE:
            import librosa
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=SAMPLE_RATE,
                res_type="polyphase",
            )

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
        raw_length = len(audio_data)
        if raw_length < self.max_length:
            audio_data = np.pad(audio_data, (0, self.max_length - raw_length))

        inputs = self.processor(
            audio_data,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        input_values = inputs.input_values.squeeze(0)

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
            "input_length": torch.tensor(raw_length, dtype=torch.long),
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


def compute_class_weights(split, encoders: dict[str, dict[str, int]]) -> dict[str, list[float]]:
    """Compute sklearn-style 'balanced' class weights per task.

    ``w_i = N / (K * n_i)`` so each class's total gradient mass ``n_i * w_i``
    equals the constant ``N / K`` — i.e. every class contributes equally to the
    loss regardless of how rare/common it is.

    Rows with missing labels (encoded as -100) are ignored in the count.
    Classes with zero samples get weight 0 (they can't be learned anyway) and
    trigger a warning.
    """
    import warnings

    weights: dict[str, list[float]] = {}
    task_to_field = {"emotion": "emotion", "gender": "gender", "age": "age_category"}
    no_audio = split.remove_columns(["audio"]) if "audio" in split.column_names else split
    for task, field in task_to_field.items():
        enc = encoders[task]
        K = len(enc)
        counts = [0] * K
        for row in no_audio:
            v = row.get(field)
            if v is None:
                continue
            s = str(v).strip()
            if not s or s not in enc:
                continue
            counts[enc[s]] += 1
        N = sum(counts)
        if N == 0:
            warnings.warn(f"compute_class_weights: task={task!r} has no labeled rows; weights=1s", stacklevel=2)
            weights[task] = [1.0] * K
            continue
        w = []
        for c in counts:
            if c == 0:
                warnings.warn(
                    f"compute_class_weights: task={task!r} has zero samples for a class; setting weight=0",
                    stacklevel=2,
                )
                w.append(0.0)
            else:
                w.append(N / (K * c))
        weights[task] = w
    return weights


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
