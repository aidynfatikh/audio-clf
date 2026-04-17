"""Shared schema for the split builder."""

from __future__ import annotations

from typing import TypedDict

DATASET_BATCH01 = "batch01"
DATASET_BATCH02 = "batch02"
DATASET_KAZEMO = "kazemo"

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"

ALL_SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)
ALL_DATASETS = (DATASET_BATCH01, DATASET_BATCH02, DATASET_KAZEMO)

MANIFEST_COLUMNS = (
    "dataset",
    "row_id",
    "source_index",
    "speaker_id",
    "emotion",
    "gender",
    "age_category",
    "augmented",
    "split",
)


class NormalizedRow(TypedDict, total=False):
    dataset: str
    row_id: str
    source_index: int
    speaker_id: str | None
    emotion: str | None
    gender: str | None
    age_category: str | None
    augmented: bool | None
