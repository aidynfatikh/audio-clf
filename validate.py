#!/usr/bin/env python3
"""
Build and evaluate a balanced non-augmented validation set.

Pipeline:
1) Download or load dataset 01gumano1d/batch01-validation-test from the Hugging Face cache (full split in memory-mapped Arrow; not streamed).
2) Keep only rows marked as non-augmented (metadata.augmented == False).
3) Stratified first-fit: up to N unique clips per (emotion, gender, age) in row order;
   stop early when every stratum has N (default N=18 → 1008).
4) If some strata are short, the script prints counts, then draws random additional
   non-augmented clips (uniform reservoir over the split) to reach 1008 unique ids
   when possible; if the split still lacks enough unique clips, it pads with replacement
   and distinct synthetic sample_id suffixes so the eval size is exactly 1008.
5) Run inference using a checkpoint from models/ (default finetuned best).
6) Save JSON reports for subset composition and evaluation metrics.

Usage:
  python validate.py
  python validate.py --model-path models/finetune/best_model_finetuned.pt
  python validate.py --samples-per-stratum 18 --batch-size 8
  python validate.py --fill-seed 123
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
os.environ["TORCHCODEC_QUIET"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Audio, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor

from inference import load_model, resolve_checkpoint
from load_data import read_audio
from train import SAMPLE_RATE


DEFAULT_DATASET_ID = "01gumano1d/batch01-validation-test"
DEFAULT_SPLIT = "train"
DEFAULT_SAMPLES_PER_STRATUM = 18
DEFAULT_BATCH_SIZE = 8


def _norm_text(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _norm_emotion(v: Any) -> str | None:
    s = _norm_text(v)
    if s is None:
        return None
    x = s.lower()
    mapping = {
        "angry": "angry",
        "disgust": "disgusted",
        "disgusted": "disgusted",
        "fear": "fearful",
        "fearful": "fearful",
        "happy": "happy",
        "neutral": "neutral",
        "sad": "sad",
        "surprise": "surprised",
        "surprised": "surprised",
    }
    return mapping.get(x, x)


def _norm_gender(v: Any) -> str | None:
    s = _norm_text(v)
    if s is None:
        return None
    x = s.lower()
    mapping = {
        "m": "M",
        "male": "M",
        "man": "M",
        "f": "F",
        "female": "F",
        "woman": "F",
    }
    return mapping.get(x, s)


def _norm_age(v: Any) -> str | None:
    s = _norm_text(v)
    if s is None:
        return None
    x = s.lower()
    mapping = {
        "adult": "adult",
        "young": "young",
        "youth": "young",
        "child": "child",
        "kid": "child",
        "children": "child",
        "senior": "senior",
        "elder": "senior",
        "elderly": "senior",
    }
    return mapping.get(x, x)


def _to_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _parse_metadata(md: Any) -> dict[str, Any]:
    if isinstance(md, dict):
        return md
    if isinstance(md, str):
        text = md.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            # Best-effort fallback for pseudo-JSON strings.
            out: dict[str, Any] = {}
            m_aug = re.search(r'"?augmented"?\s*[:=]\s*(true|false)', text, flags=re.I)
            if m_aug:
                out["augmented"] = m_aug.group(1).lower() == "true"
            return out
    return {}


def _is_non_augmented(row: dict[str, Any]) -> bool:
    # Strict policy: include only rows with explicit augmented=False.
    for key in ("augmented", "is_augmented"):
        if key in row:
            b = _to_bool(row.get(key))
            if b is not None:
                return b is False

    md = _parse_metadata(row.get("metadata"))
    if "augmented" in md:
        b = _to_bool(md.get("augmented"))
        if b is not None:
            return b is False

    return False


def _row_label(row: dict[str, Any], primary_keys: tuple[str, ...], md_keys: tuple[str, ...], norm_fn):
    for k in primary_keys:
        if k in row:
            val = norm_fn(row.get(k))
            if val is not None:
                return val

    md = _parse_metadata(row.get("metadata"))
    for k in md_keys:
        if k in md:
            val = norm_fn(md.get(k))
            if val is not None:
                return val

    return None


def _extract_labels(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    emotion = _row_label(row, ("emotion", "Emotion"), ("emotion", "Emotion"), _norm_emotion)
    gender = _row_label(row, ("gender", "Gender", "sex"), ("gender", "Gender", "sex"), _norm_gender)
    age = _row_label(
        row,
        ("age_category", "age_group", "age", "Age"),
        ("age_category", "age_group", "age", "Age"),
        _norm_age,
    )
    return emotion, gender, age


@dataclass
class BuildStats:
    scanned_rows: int = 0
    non_aug_rows: int = 0
    valid_rows: int = 0
    missing_label_rows: int = 0
    out_of_target_rows: int = 0
    missing_audio_rows: int = 0


class SelectedAudioDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        processor: Wav2Vec2FeatureExtractor,
        emotion_encoder: dict[str, int],
        gender_encoder: dict[str, int],
        age_encoder: dict[str, int],
        max_length: int = 160000,
    ):
        self.rows = rows
        self.processor = processor
        self.emotion_encoder = emotion_encoder
        self.gender_encoder = gender_encoder
        self.age_encoder = age_encoder
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        audio_data, sr = read_audio(row["audio"])

        if sr != SAMPLE_RATE:
            import librosa

            audio_data = librosa.resample(
                audio_data.astype(np.float32),
                orig_sr=sr,
                target_sr=SAMPLE_RATE,
            )

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

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
            attention_mask = torch.ones_like(input_values, dtype=torch.long)

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "emotion": torch.tensor(self.emotion_encoder[row["emotion"]], dtype=torch.long),
            "gender": torch.tensor(self.gender_encoder[row["gender"]], dtype=torch.long),
            "age": torch.tensor(self.age_encoder[row["age_category"]], dtype=torch.long),
            "sample_id": row.get("sample_id", f"row-{idx}"),
        }


def _print_stratum_inventory(
    *,
    target: dict[tuple[str, str, str], int],
    stratum_universe: dict[tuple[str, str, str], set[str]],
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]],
    requested_per_stratum: int,
) -> None:
    """Print unique non-augmented counts per stratum and max balanced subset size."""
    rows_out = []
    for key in sorted(target.keys()):
        sk = "::".join(key)
        u = len(stratum_universe.get(key, ()))
        picked = len(buckets.get(key, ()))
        rows_out.append((sk, u, picked, target[key]))
    max_per_stratum = min(u for _, u, _, _ in rows_out) if rows_out else 0
    n_strata = len(rows_out)
    max_balanced_total = max_per_stratum * n_strata

    print("\n[build] Non-augmented UNIQUE clips per (emotion::gender::age) after full scan:", flush=True)
    print(f"  {'stratum':<48} {'uniq_in_split':>14} {'picked':>8} {'requested':>10}", flush=True)
    for sk, u, picked, req in rows_out:
        print(f"  {sk:<48} {u:>14} {picked:>8} {req:>10}", flush=True)
    print(
        f"\n[build] Largest balanced grid this split allows: "
        f"{max_per_stratum} per stratum × {n_strata} strata = {max_balanced_total} total "
        f"(requires every stratum to have at least {max_per_stratum} unique clips).",
        flush=True,
    )
    shortfalls = [(sk, u, req) for sk, u, picked, req in rows_out if u < req]
    if shortfalls:
        print("\n[build] Strata below requested count (unique_in_split < requested):", flush=True)
        for sk, u, req in sorted(shortfalls, key=lambda t: (t[1], t[0])):
            print(f"  {sk}: have {u}, need {req}", flush=True)


def _reservoir_fill_rows(
    dataset,
    *,
    k: int,
    exclude_ids: set[str],
    target_emotions: list[str],
    target_genders: list[str],
    target_ages: list[str],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int]:
    """Up to k uniform random eligible non-augmented rows excluding exclude_ids. Returns (rows, eligible_seen)."""
    reservoir: list[dict[str, Any]] = []
    n_seen = 0
    for idx, row in enumerate(dataset):
        if not _is_non_augmented(row):
            continue
        emotion, gender, age = _extract_labels(row)
        if emotion is None or gender is None or age is None:
            continue
        if emotion not in target_emotions or gender not in target_genders or age not in target_ages:
            continue
        if "audio" not in row:
            continue
        sample_id = str(row.get("id", row.get("uid", f"row-{idx}")))
        if sample_id in exclude_ids:
            continue
        n_seen += 1
        row_copy = dict(row)
        row_copy["emotion"] = emotion
        row_copy["gender"] = gender
        row_copy["age_category"] = age
        row_copy["sample_id"] = sample_id
        if len(reservoir) < k:
            reservoir.append(row_copy)
        elif rng.random() < k / n_seen:
            reservoir[rng.randint(0, k - 1)] = row_copy
    return reservoir, n_seen


def _build_balanced_subset(
    dataset,
    *,
    samples_per_stratum: int,
    target_emotions: list[str],
    target_genders: list[str],
    target_ages: list[str],
    log_every: int = 500,
    fill_seed: int = 42,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any], BuildStats]:
    target = {
        (e, g, a): samples_per_stratum
        for e in target_emotions
        for g in target_genders
        for a in target_ages
    }
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    stratum_universe: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    global_sample_ids: set[str] = set()
    stats = BuildStats()
    total_needed = len(target) * samples_per_stratum
    t0 = time.time()

    print(
        f"[build] Sequential first-fit: up to {samples_per_stratum} unique clips per stratum, "
        f"stop when all strata full. Progress every {log_every} rows (0 = only start/done).",
        flush=True,
    )

    def all_strata_full() -> bool:
        return all(len(buckets[k]) >= samples_per_stratum for k in target)

    stopped_early_all_full = False
    for idx, row in enumerate(dataset):
        stats.scanned_rows += 1

        if not _is_non_augmented(row):
            continue
        stats.non_aug_rows += 1

        emotion, gender, age = _extract_labels(row)
        if emotion is None or gender is None or age is None:
            stats.missing_label_rows += 1
            continue

        if emotion not in target_emotions or gender not in target_genders or age not in target_ages:
            stats.out_of_target_rows += 1
            continue

        key = (emotion, gender, age)
        if "audio" not in row:
            stats.missing_audio_rows += 1
            continue

        sample_id = str(row.get("id", row.get("uid", f"row-{idx}")))
        if sample_id in stratum_universe[key]:
            continue
        stratum_universe[key].add(sample_id)

        if sample_id in global_sample_ids:
            continue

        stats.valid_rows += 1
        row_copy = dict(row)
        row_copy["emotion"] = emotion
        row_copy["gender"] = gender
        row_copy["age_category"] = age
        row_copy["sample_id"] = sample_id

        if len(buckets[key]) < target[key]:
            buckets[key].append(row_copy)
            global_sample_ids.add(sample_id)

        if log_every > 0 and (stats.scanned_rows == 1 or stats.scanned_rows % log_every == 0):
            filled = sum(len(v) for v in buckets.values())
            elapsed = max(time.time() - t0, 1e-9)
            rate = stats.scanned_rows / elapsed
            print(
                "[build] scanned={} non_aug={} eligible={} slots={}/{} ({:.1f}%) rate={:.1f} rows/s elapsed={:.1f}s".format(
                    stats.scanned_rows,
                    stats.non_aug_rows,
                    stats.valid_rows,
                    filled,
                    total_needed,
                    100.0 * filled / max(total_needed, 1),
                    rate,
                    elapsed,
                ),
                flush=True,
            )

        if all_strata_full():
            stopped_early_all_full = True
            break

    filled = sum(len(v) for v in buckets.values())
    elapsed = max(time.time() - t0, 1e-9)
    print(
        "[build] done: scanned={} non_aug={} eligible={} slots={}/{} ({:.1f}%) elapsed={:.1f}s".format(
            stats.scanned_rows,
            stats.non_aug_rows,
            stats.valid_rows,
            filled,
            total_needed,
            100.0 * filled / max(total_needed, 1),
            elapsed,
        ),
        flush=True,
    )

    per_stratum_universe = { "::".join(k): len(stratum_universe[k]) for k in sorted(target.keys())}
    per_stratum_picked = { "::".join(k): len(buckets[k]) for k in sorted(target.keys())}
    max_per_stratum = min(per_stratum_universe.values()) if per_stratum_universe else 0
    n_strata = len(target)
    max_balanced_total = max_per_stratum * n_strata

    zero = any(len(stratum_universe[k]) == 0 for k in target)
    short = any(len(buckets[k]) < target[k] for k in target)

    selected: list[dict[str, Any]] = []
    for key in sorted(target.keys()):
        selected.extend(buckets[key][: target[key]])

    stratified_n = len(selected)
    rng = random.Random(fill_seed)
    random_fill_unique = 0
    random_fill_repad = 0

    if zero or short:
        print("\n[build] Imperfect stratification; summarizing non-augmented inventory:", flush=True)
        _print_stratum_inventory(
            target=target, stratum_universe=stratum_universe, buckets=buckets, requested_per_stratum=samples_per_stratum
        )
        print(
            f"\n[build] Ideal full grid without top-up would need at least {max_per_stratum} per stratum "
            f"({max_balanced_total} total). Top-up will add random non-augmented clips to reach {total_needed}.",
            flush=True,
        )

    need = total_needed - len(selected)
    if need > 0:
        print(f"[build] Top-up pass: sampling {need} clips (uniform reservoir, seed={fill_seed})...", flush=True)
        exclude = set(global_sample_ids)
        extras, pool_n = _reservoir_fill_rows(
            dataset,
            k=need,
            exclude_ids=exclude,
            target_emotions=target_emotions,
            target_genders=target_genders,
            target_ages=target_ages,
            rng=rng,
        )
        for row in extras:
            global_sample_ids.add(row["sample_id"])
        selected.extend(extras)
        random_fill_unique = len(extras)
        need2 = total_needed - len(selected)
        if need2 > 0:
            pad_base = list(selected)
            if not pad_base:
                print("[build] Fatal: no clips available to pad eval set.", flush=True)
                return None, {
                    "success": False,
                    "reason": "empty_pad_pool",
                    "per_stratum_unique_in_split": per_stratum_universe,
                    "per_stratum_picked": per_stratum_picked,
                }, stats
            print(
                f"[build] Only {pool_n} unique top-up candidates for {need} slots; "
                f"padding {need2} rows by resampling (duplicate audio, new sample_id suffix).",
                flush=True,
            )
            while len(selected) < total_needed:
                base = dict(rng.choice(pad_base))
                sid = str(base.get("sample_id", "unknown"))
                base["sample_id"] = f"{sid}__repad_{len(selected)}"
                selected.append(base)
                random_fill_repad += 1

    if len(selected) != total_needed:
        print(f"[build] Fatal: expected {total_needed} rows, have {len(selected)}.", flush=True)
        return None, {"success": False, "reason": "size_mismatch", "have": len(selected)}, stats

    counts: dict[str, Any] = {
        "success": True,
        "sampling": "stratified_first_fit_plus_random_topup",
        "fill_seed": fill_seed,
        "stratified_count": stratified_n,
        "random_topup_unique_count": random_fill_unique,
        "random_topup_repad_count": random_fill_repad,
        "imperfect_stratification": bool(zero or short),
        "partial_scan": stopped_early_all_full,
        "by_stratum_picked": per_stratum_picked,
        "by_stratum_final": {"::".join(k): target[k] for k in sorted(target.keys())},
        "total": len(selected),
        "upsampled_total": random_fill_repad,
    }
    if stopped_early_all_full:
        counts["by_stratum_unique_seen_before_stop"] = per_stratum_universe
        counts["note_partial_universe"] = (
            "Scan stopped when all strata were full; per-stratum unique counts are lower bounds."
        )
    else:
        counts["by_stratum_unique_in_split"] = per_stratum_universe
        counts["max_balanced_per_stratum"] = max_per_stratum
        counts["max_balanced_total"] = max_balanced_total
    return selected, counts, stats


def _metrics_from_predictions(
    y_true: list[int],
    y_pred: list[int],
    id2label: dict[int, str],
) -> dict[str, Any]:
    n = len(y_true)
    labels = sorted(id2label.keys())

    confusion = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n > 0 else 0.0

    per_class_accuracy: dict[str, float] = {}
    support: dict[str, int] = {}

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for i, lab in enumerate(labels):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(len(labels))) - tp
        fn = sum(confusion[i][c] for c in range(len(labels))) - tp
        sup = sum(confusion[i])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        per_class_accuracy[id2label[lab]] = recall
        support[id2label[lab]] = sup

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "support_total": n,
        "per_class_accuracy": per_class_accuracy,
        "support": support,
        "confusion_matrix": confusion,
        "confusion_labels": [id2label[i] for i in labels],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Balanced non-augmented validation builder + evaluator")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--samples-per-stratum", type=int, default=DEFAULT_SAMPLES_PER_STRATUM)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Print subset-build progress every N scanned rows (also prints on row 1). Use 0 for quiet except start/done.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "batch01-validation-test")
    parser.add_argument("--fill-seed", type=int, default=42, help="RNG seed for random top-up sampling (default: 42)")
    parser.add_argument("--out-json", type=Path, default=Path(__file__).resolve().parent / "eval" / "validation_eval_results.json")
    parser.add_argument("--subset-json", type=Path, default=Path(__file__).resolve().parent / "eval" / "validation_subset_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent
    load_dotenv(repo_root / ".env")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.model_path if args.model_path is not None else resolve_checkpoint()

    (
        model,
        emotion_encoder,
        gender_encoder,
        age_encoder,
        id2emotion,
        id2gender,
        id2age,
        processor,
    ) = load_model(ckpt_path=ckpt, device=device)

    target_emotions = [k for k, _ in sorted(emotion_encoder.items(), key=lambda kv: kv[1])]
    target_genders = [k for k, _ in sorted(gender_encoder.items(), key=lambda kv: kv[1])]
    target_ages = [k for k, _ in sorted(age_encoder.items(), key=lambda kv: kv[1])]

    expected_total = args.samples_per_stratum * len(target_emotions) * len(target_genders) * len(target_ages)

    print(f"Checkpoint: {ckpt}")
    print(f"Device: {device}")
    print(f"Target classes: emotions={target_emotions}, genders={target_genders}, ages={target_ages}")
    print(f"Target sample count: {args.samples_per_stratum} x {len(target_emotions)} x {len(target_genders)} x {len(target_ages)} = {expected_total}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[load] Loading dataset (download or HF cache) → {args.cache_dir}: "
        f"{args.dataset_id!r} split={args.split!r}",
        flush=True,
    )
    ds = load_dataset(args.dataset_id, cache_dir=str(args.cache_dir))
    split = ds[args.split] if args.split in ds else ds[list(ds.keys())[0]]
    if "audio" in split.column_names:
        split = split.cast_column("audio", Audio(decode=False))
    print(f"[load] Ready: {len(split):,} rows. Building balanced non-augmented subset...", flush=True)
    dataset_iter = split

    selected_rows, subset_counts, build_stats = _build_balanced_subset(
        dataset_iter,
        samples_per_stratum=args.samples_per_stratum,
        target_emotions=target_emotions,
        target_genders=target_genders,
        target_ages=target_ages,
        log_every=args.log_every,
        fill_seed=args.fill_seed,
    )

    if selected_rows is None:
        args.subset_json.parent.mkdir(parents=True, exist_ok=True)
        fail_report = {
            "dataset_id": args.dataset_id,
            "split": args.split,
            "non_augmented_only": True,
            "samples_per_stratum_requested": args.samples_per_stratum,
            "target_classes": {
                "emotion": target_emotions,
                "gender": target_genders,
                "age": target_ages,
            },
            "subset_build_failed": True,
            **subset_counts,
            "build_stats": {
                "scanned_rows": build_stats.scanned_rows,
                "non_augmented_rows": build_stats.non_aug_rows,
                "eligible_rows": build_stats.valid_rows,
                "missing_label_rows": build_stats.missing_label_rows,
                "out_of_target_rows": build_stats.out_of_target_rows,
                "missing_audio_rows": build_stats.missing_audio_rows,
            },
        }
        with open(args.subset_json, "w") as f:
            json.dump(fail_report, f, indent=2)
        print("[build] Wrote failure inventory to", args.subset_json, flush=True)
        sys.exit(1)

    eval_dataset = SelectedAudioDataset(
        rows=selected_rows,
        processor=processor,
        emotion_encoder=emotion_encoder,
        gender_encoder=gender_encoder,
        age_encoder=age_encoder,
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model.eval()

    y_true = {"emotion": [], "gender": [], "age": []}
    y_pred = {"emotion": [], "gender": [], "age": []}
    sample_outputs: list[dict[str, Any]] = []

    print(f"[eval] Running model inference on {len(eval_dataset)} samples (batch_size={args.batch_size})", flush=True)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            emotion_logits, gender_logits, age_logits = model(input_values, attention_mask=attention_mask)

            emotion_pred = emotion_logits.argmax(dim=1).cpu().tolist()
            gender_pred = gender_logits.argmax(dim=1).cpu().tolist()
            age_pred = age_logits.argmax(dim=1).cpu().tolist()

            emotion_true = batch["emotion"].cpu().tolist()
            gender_true = batch["gender"].cpu().tolist()
            age_true = batch["age"].cpu().tolist()

            y_true["emotion"].extend(emotion_true)
            y_true["gender"].extend(gender_true)
            y_true["age"].extend(age_true)

            y_pred["emotion"].extend(emotion_pred)
            y_pred["gender"].extend(gender_pred)
            y_pred["age"].extend(age_pred)

            emo_conf = F.softmax(emotion_logits, dim=1).max(dim=1).values.cpu().tolist()
            gen_conf = F.softmax(gender_logits, dim=1).max(dim=1).values.cpu().tolist()
            age_conf = F.softmax(age_logits, dim=1).max(dim=1).values.cpu().tolist()

            sample_ids = batch["sample_id"]
            for i in range(len(sample_ids)):
                sample_outputs.append(
                    {
                        "sample_id": str(sample_ids[i]),
                        "emotion_true": id2emotion[emotion_true[i]],
                        "emotion_pred": id2emotion[emotion_pred[i]],
                        "emotion_confidence": round(float(emo_conf[i]), 6),
                        "gender_true": id2gender[gender_true[i]],
                        "gender_pred": id2gender[gender_pred[i]],
                        "gender_confidence": round(float(gen_conf[i]), 6),
                        "age_true": id2age[age_true[i]],
                        "age_pred": id2age[age_pred[i]],
                        "age_confidence": round(float(age_conf[i]), 6),
                    }
                )

    metrics = {
        "emotion": _metrics_from_predictions(y_true["emotion"], y_pred["emotion"], id2emotion),
        "gender": _metrics_from_predictions(y_true["gender"], y_pred["gender"], id2gender),
        "age": _metrics_from_predictions(y_true["age"], y_pred["age"], id2age),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.subset_json.parent.mkdir(parents=True, exist_ok=True)

    subset_manifest = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "non_augmented_only": True,
        "samples_per_stratum": args.samples_per_stratum,
        "validation_sample_ids": [str(r.get("sample_id", "")) for r in selected_rows],
        "target_classes": {
            "emotion": target_emotions,
            "gender": target_genders,
            "age": target_ages,
        },
        "expected_total": expected_total,
        "actual_total": len(selected_rows),
        "build_stats": {
            "scanned_rows": build_stats.scanned_rows,
            "non_augmented_rows": build_stats.non_aug_rows,
            "eligible_rows": build_stats.valid_rows,
            "missing_label_rows": build_stats.missing_label_rows,
            "out_of_target_rows": build_stats.out_of_target_rows,
            "missing_audio_rows": build_stats.missing_audio_rows,
        },
        "counts": subset_counts,
    }

    eval_report = {
        "checkpoint": str(ckpt),
        "device": str(device),
        "dataset_id": args.dataset_id,
        "split": args.split,
        "non_augmented_only": True,
        "samples_per_stratum": args.samples_per_stratum,
        "total_samples": len(selected_rows),
        "metrics": metrics,
        "samples": sample_outputs,
    }

    with open(args.subset_json, "w") as f:
        json.dump(subset_manifest, f, indent=2)

    with open(args.out_json, "w") as f:
        json.dump(eval_report, f, indent=2)

    print("Saved subset manifest:", args.subset_json)
    print("Saved evaluation report:", args.out_json)
    print(
        "Accuracies:",
        f"emotion={metrics['emotion']['accuracy']:.4f}",
        f"gender={metrics['gender']['accuracy']:.4f}",
        f"age={metrics['age']['accuracy']:.4f}",
    )


if __name__ == "__main__":
    main()
