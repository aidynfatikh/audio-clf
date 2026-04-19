"""Stratified, group-disjoint 3-way split (train/val/test).

Wraps sklearn.model_selection.StratifiedGroupKFold. Caller supplies rows, a
stratification field (emotion, or joint "emotion+gender+age"), and a grouping
field (speaker_id). Rows with a missing group fall back to a unique
per-row group, which is equivalent to plain stratification for those rows.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from splits.schema import (
    NormalizedRow,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)


def _build_stratify_key(row: NormalizedRow, stratify_by: str) -> str:
    if stratify_by == "emotion":
        return str(row.get("emotion") or "__none__")
    if stratify_by == "emotion+gender+age":
        return (
            f"{row.get('emotion') or '_e'}|"
            f"{row.get('gender') or '_g'}|"
            f"{row.get('age_category') or '_a'}"
        )
    raise ValueError(f"Unsupported stratify_by={stratify_by!r}")


def _build_group_key(row: NormalizedRow, row_idx: int, group_by_field: str) -> str:
    val = row.get(group_by_field)
    if val is None or (isinstance(val, str) and not val.strip()):
        # Singleton group — each row is its own group.
        return f"__singleton__{row_idx}"
    return str(val)


def _pick_best_fold(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    target_frac: float,
    seed: int,
    forbidden: set[int] | None = None,
) -> tuple[set[int], int]:
    """Run StratifiedGroupKFold(k = round(1/target_frac)) and return the fold
    whose size is closest to target_frac * n. Returns (indices_set, k_used)."""
    n = len(x)
    k = max(int(round(1.0 / max(target_frac, 1e-6))), 3)
    n_groups = len({tuple() if g is None else g for g in groups.tolist()})
    k = min(k, n_groups) if n_groups >= 2 else 2
    if k < 2:
        raise ValueError(f"Not enough groups ({n_groups}) to split")

    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = list(skf.split(x, y, groups))

    target_count = target_frac * n
    forbidden = forbidden or set()

    best_idx = None
    best_diff = None
    for i, (_, test_arr) in enumerate(folds):
        test_set = set(test_arr.tolist())
        if forbidden and (test_set & forbidden):
            continue
        diff = abs(len(test_set) - target_count)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_idx = test_set
    if best_idx is None:
        # All folds touch forbidden rows (possible when groups span them).
        # Fall back to smallest-diff fold even if it overlaps; caller will dedup.
        for _, test_arr in folds:
            test_set = set(test_arr.tolist())
            diff = abs(len(test_set) - target_count)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = test_set
    return best_idx or set(), k


def stratified_grouped_three_way(
    rows: list[NormalizedRow],
    ratios: dict[str, float],
    stratify_by: str,
    group_by_field: str,
    seed: int,
) -> dict[str, list[int]]:
    """Produce train/val/test index lists into ``rows``.

    Strategy: nested StratifiedGroupKFold.
      1. Peel off test: run with k = round(1/test_ratio), pick fold closest to target.
      2. On remaining rows, peel off val: run with k = round(1/(val_ratio/(1-test_ratio))),
         pick fold closest to target. Because groups are preserved across both passes,
         the three splits remain speaker-disjoint (a group removed in pass 1 cannot
         reappear in pass 2).
    """
    if not rows:
        return {SPLIT_TRAIN: [], SPLIT_VAL: [], SPLIT_TEST: []}

    n = len(rows)
    y_full = np.array([_build_stratify_key(r, stratify_by) for r in rows])
    groups_full = np.array([_build_group_key(r, i, group_by_field) for i, r in enumerate(rows)])
    x_full = np.zeros(n, dtype=np.int8)

    test_ratio = float(ratios.get(SPLIT_TEST, 0.15))
    val_ratio = float(ratios.get(SPLIT_VAL, 0.15))
    if test_ratio <= 0 or val_ratio <= 0:
        raise ValueError(f"val and test ratios must be > 0, got {ratios}")

    # Pass 1 — test
    test_idx, _ = _pick_best_fold(
        x_full, y_full, groups_full, target_frac=test_ratio, seed=seed
    )

    # Pass 2 — val on remaining rows
    remaining = [i for i in range(n) if i not in test_idx]
    if not remaining:
        train_idx = []
        val_idx: set[int] = set()
    else:
        rem_idx = np.array(remaining, dtype=np.int64)
        x_r = x_full[rem_idx]
        y_r = y_full[rem_idx]
        g_r = groups_full[rem_idx]
        val_frac_in_rem = val_ratio / max(1.0 - test_ratio, 1e-6)
        val_local, _ = _pick_best_fold(
            x_r, y_r, g_r, target_frac=val_frac_in_rem, seed=seed + 1
        )
        val_idx = {int(rem_idx[i]) for i in val_local}
        train_idx = [i for i in remaining if i not in val_idx]

    # Final sanity: no overlap across the three buckets
    if val_idx & test_idx:
        # Should be impossible by construction; dedup safely.
        val_idx = val_idx - test_idx
    return {
        SPLIT_TRAIN: sorted(train_idx),
        SPLIT_VAL: sorted(val_idx),
        SPLIT_TEST: sorted(test_idx),
    }
