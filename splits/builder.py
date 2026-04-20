"""Split builder: YAML config → parquet manifests + summary.json."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from huggingface_hub import login as hf_login

from splits.io import build_summary, write_manifests, write_summary
from splits.kazemo_speakers import kazemo_three_way_split, resolve_kazemo_speakers
from splits.schema import (
    ALL_DATASETS,
    ALL_SPLITS,
    DATASET_KAZEMO,
    NormalizedRow,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from splits.sources import _resolve_aug_policy, load_corpus
from splits.stratified_group import stratified_grouped_three_way

REPO_ROOT = Path(__file__).resolve().parent.parent


def _hf_login_if_needed() -> None:
    load_dotenv(REPO_ROOT / ".env")
    tok = os.environ.get("HF_TOKEN")
    if tok:
        try:
            hf_login(token=tok)
        except Exception as e:
            print(f"[builder] hf_login failed (continuing): {e}")


def _load_yaml(path: Path) -> tuple[dict[str, Any], str]:
    text = Path(path).read_text()
    cfg = yaml.safe_load(text) or {}
    return cfg, text


def _resolve_cache_dir(raw: str | None, default: str) -> Path:
    if not raw:
        return REPO_ROOT / default
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _assign_train_only(rows: list[NormalizedRow]) -> dict[str, list[int]]:
    return {SPLIT_TRAIN: list(range(len(rows))), SPLIT_VAL: [], SPLIT_TEST: []}


def _check_speaker_disjoint(assignments: dict[str, list[NormalizedRow]]) -> list[str]:
    """Return ``"<dataset>:<speaker>"`` entries that leak across splits.

    Per-dataset rules:
      * train ↔ val and train ↔ test overlap is *never* allowed.
      * val ↔ test overlap is allowed only for KazEmo (only 3 speakers exist,
        so one speaker is intentionally shared between val and test).
    """
    per_ds: dict[str, dict[str, set[str]]] = {}
    for split, rows in assignments.items():
        for r in rows:
            spk = r.get("speaker_id")
            if not spk:
                continue
            ds = r["dataset"]
            buckets = per_ds.setdefault(ds, {s: set() for s in ALL_SPLITS})
            buckets[split].add(spk)

    leaked: list[str] = []
    for ds, buckets in per_ds.items():
        train_val = buckets[SPLIT_TRAIN] & buckets[SPLIT_VAL]
        train_test = buckets[SPLIT_TRAIN] & buckets[SPLIT_TEST]
        for spk in sorted(train_val | train_test):
            leaked.append(f"{ds}:{spk}")
        if ds != DATASET_KAZEMO:
            for spk in sorted(buckets[SPLIT_VAL] & buckets[SPLIT_TEST]):
                leaked.append(f"{ds}:{spk}")
    return leaked


_ALLOWED_AUG_SPLITS = {
    "drop_all": set(),
    "train_only": {SPLIT_TRAIN},
    "train_val_only": {SPLIT_TRAIN, SPLIT_VAL},
    "keep_all": {SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST},
}


def _compute_leak_check(
    assignments: dict[str, list[NormalizedRow]],
    corpora_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Verify every aug row sits in a split its per-dataset policy permits.

    Returns per-split and per-corpus aug counts plus a boolean `passed`.
    `passed=False` means a real sample's augmented sibling landed in a split
    the policy forbids — unsafe manifests, caller should abort.
    """
    # Resolve per-dataset allowed splits from configured aug policy.
    ds_allowed: dict[str, set[str]] = {}
    for ds, ds_cfg in (corpora_cfg or {}).items():
        ds_cfg = ds_cfg or {}
        try:
            pol = _resolve_aug_policy(ds_cfg)
        except Exception:
            pol = "drop_all"
        ds_allowed[ds] = _ALLOWED_AUG_SPLITS.get(pol, set())

    per_split_aug: dict[str, int] = {s: 0 for s in ALL_SPLITS}
    per_corpus_aug: dict[str, dict[str, int]] = {}
    violations: list[str] = []
    for split, rows in assignments.items():
        for r in rows:
            if r.get("augmented") is not True:
                continue
            ds = r["dataset"]
            per_split_aug[split] += 1
            bucket = per_corpus_aug.setdefault(ds, {s: 0 for s in ALL_SPLITS})
            bucket[split] += 1
            allowed = ds_allowed.get(ds, {SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST})
            if split not in allowed:
                violations.append(f"{ds}:{split}:{r.get('row_id')}")

    return {
        "passed": not violations,
        "per_split_aug_count": per_split_aug,
        "per_corpus_aug_count": per_corpus_aug,
        "policy_per_corpus": {
            ds: sorted(splits) for ds, splits in ds_allowed.items()
        },
        "violations": violations[:20],
        "total_violations": len(violations),
    }


def _check_row_id_disjoint(assignments: dict[str, list[NormalizedRow]]) -> list[tuple[str, str]]:
    seen: dict[tuple[str, str], str] = {}
    dupes: list[tuple[str, str]] = []
    for split, rows in assignments.items():
        for r in rows:
            key = (r["dataset"], r["row_id"])
            if key in seen and seen[key] != split:
                dupes.append(key)
            else:
                seen[key] = split
    return dupes


def _check_emotion_coverage(
    assignments: dict[str, list[NormalizedRow]], min_per_split: int
) -> list[str]:
    errs: list[str] = []
    for split in ALL_SPLITS:
        counts: dict[str, int] = {}
        for r in assignments.get(split, []):
            emo = r.get("emotion")
            if emo:
                counts[emo] = counts.get(emo, 0) + 1
        for emo, n in counts.items():
            if n < min_per_split:
                errs.append(f"{split}:{emo}:{n}<{min_per_split}")
    return errs


def _check_ratio_drift(
    assignments: dict[str, list[NormalizedRow]],
    cfg: dict[str, Any],
    tolerance_pp: float,
) -> list[str]:
    """Check per-corpus ratios are within tolerance of target, for 'split' mode corpora."""
    errs: list[str] = []
    corpora = cfg.get("corpora", {})
    for ds, ds_cfg in corpora.items():
        ds_cfg = ds_cfg or {}
        if _normalize_mode(ds_cfg.get("mode")) != "split":
            continue
        ratios = ds_cfg.get("ratios", {})
        if not ratios:
            continue
        # Count this corpus per split
        counts = {s: 0 for s in ALL_SPLITS}
        for s in ALL_SPLITS:
            for r in assignments.get(s, []):
                if r["dataset"] == ds:
                    counts[s] += 1
        total = sum(counts.values())
        if total == 0:
            continue
        for s, target in ratios.items():
            actual = counts[s] / total
            drift = abs(actual - float(target)) * 100.0
            if drift > tolerance_pp:
                errs.append(f"{ds}:{s} drift={drift:.1f}pp (actual={actual:.3f}, target={target})")
    return errs


def _normalize_mode(raw: Any) -> str:
    """YAML 1.1 parses bare ``off`` as ``False``. Normalize to a string token."""
    if raw is False or raw is None:
        return "off"
    if raw is True:
        return "split"
    return str(raw).strip().lower() or "off"


def _split_corpus(
    name: str,
    ds_cfg: dict[str, Any],
    rows: list[NormalizedRow],
    *,
    global_stratify_by: str,
    global_seed: int,
    speaker_disjoint: bool,
) -> tuple[dict[str, list[int]], dict[str, Any] | None, dict[str, int]]:
    """Return (indices_per_split, kazemo_resolved_if_any, builder_drop_stats)."""
    mode = _normalize_mode(ds_cfg.get("mode"))
    if mode == "off":
        return {SPLIT_TRAIN: [], SPLIT_VAL: [], SPLIT_TEST: []}, None, {}
    if mode == "train_only":
        return _assign_train_only(rows), None, {}
    if mode != "split":
        raise ValueError(f"{name}: unknown mode={mode!r}")

    if name == DATASET_KAZEMO:
        sel = ds_cfg.get("speaker_selection", {})
        all_speakers = sorted({r["speaker_id"] for r in rows if r.get("speaker_id")})
        train_spk, valtest_spk = resolve_kazemo_speakers(
            all_speakers,
            strategy=sel.get("strategy", "by_index"),
            train_indices=sel.get("train_indices"),
            valtest_index=sel.get("valtest_index"),
            train_names=sel.get("train_names"),
            valtest_name=sel.get("valtest_name"),
        )
        valtest_ratio = ds_cfg.get("valtest_ratio", {SPLIT_VAL: 0.5, SPLIT_TEST: 0.5})
        idx = kazemo_three_way_split(
            rows,
            train_speakers=train_spk,
            valtest_speakers=valtest_spk,
            valtest_ratio=valtest_ratio,
            seed=ds_cfg.get("seed", global_seed),
        )
        resolved = {
            "discovered_speakers": all_speakers,
            "train_speakers": sorted(train_spk),
            "valtest_speakers": sorted(valtest_spk),
        }
        return idx, resolved, {}

    # batch01/batch02 — stratified grouped
    stratify_by = ds_cfg.get("stratify_by", global_stratify_by)
    group_field = "speaker_id" if speaker_disjoint else "row_id"
    ratios = ds_cfg.get("ratios", {SPLIT_TRAIN: 0.7, SPLIT_VAL: 0.15, SPLIT_TEST: 0.15})

    policy = _resolve_aug_policy(ds_cfg)
    if policy == "keep_all":
        # Split every row 3-way — augmented copies can end up anywhere.
        idx = stratified_grouped_three_way(
            rows,
            ratios=ratios,
            stratify_by=stratify_by,
            group_by_field=group_field,
            seed=ds_cfg.get("seed", global_seed),
        )
        return idx, None, {"augmented_policy": policy}

    # drop_all / train_only — sources.py has already filtered aug rows for
    # drop_all. For train_only, rows here contain both aug and non-aug; we
    # split ONLY the non-aug pool and route aug rows to train afterwards.
    non_aug_idx = [i for i, r in enumerate(rows) if r.get("augmented") is not True]
    aug_idx = [i for i, r in enumerate(rows) if r.get("augmented") is True]

    non_aug_rows = [rows[i] for i in non_aug_idx]
    core = stratified_grouped_three_way(
        non_aug_rows,
        ratios=ratios,
        stratify_by=stratify_by,
        group_by_field=group_field,
        seed=ds_cfg.get("seed", global_seed),
    )
    # Map the non-aug-local indices back to original `rows` indices.
    idx: dict[str, list[int]] = {
        split: [non_aug_idx[j] for j in core[split]] for split in core
    }

    drop_stats: dict[str, int] = {"augmented_policy": policy}
    if policy in ("train_only", "train_val_only") and aug_idx:
        allowed = {SPLIT_TRAIN} if policy == "train_only" else {SPLIT_TRAIN, SPLIT_VAL}
        route_stats = _route_augmented_rows(
            name, rows, aug_idx, idx, allowed_splits=allowed, policy=policy,
        )
        drop_stats.update(route_stats)
    elif policy == "drop_all":
        # Rows with augmented=True were already filtered at source.
        drop_stats["note"] = "aug rows filtered at source"

    return idx, None, drop_stats


def _route_augmented_rows(
    name: str,
    rows: list[NormalizedRow],
    aug_idx: list[int],
    idx: dict[str, list[int]],
    *,
    allowed_splits: set[str],
    policy: str,
) -> dict[str, int]:
    """Route augmented rows to the split their parent-group-key landed in.

    An aug row's group key (stored as `speaker_id` on NormalizedRow — this is a
    speaker for batch01 and a source_recording for batch02) is looked up
    against the non-aug assignment. If that split is in `allowed_splits` the
    aug row goes there; otherwise it is dropped. Rows with no group key are
    dropped as unattributed (we can't prove they don't leak).

    Guarantees no aug sibling of a row in a disallowed split ever appears in
    any kept split — i.e. if test is disallowed, no aug copy of a test sample
    can be in train or val.
    """
    # group_key → split lookup built from the non-aug assignment.
    group_to_split: dict[str, str] = {}
    for split in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
        for i in idx[split]:
            gk = rows[i].get("speaker_id")
            if gk:
                group_to_split[gk] = split

    added_per_split: dict[str, int] = {s: 0 for s in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)}
    unknown_group = 0
    dropped_disallowed = 0
    for i in aug_idx:
        gk = rows[i].get("speaker_id")
        if gk is None:
            unknown_group += 1
            continue
        target = group_to_split.get(gk)
        if target is None:
            # Group key never appeared in non-aug splits → parent was dropped
            # (e.g. emotion missing). Route to train if train is allowed, else drop.
            if SPLIT_TRAIN in allowed_splits:
                idx[SPLIT_TRAIN].append(i)
                added_per_split[SPLIT_TRAIN] += 1
            else:
                dropped_disallowed += 1
            continue
        if target in allowed_splits:
            idx[target].append(i)
            added_per_split[target] += 1
        else:
            dropped_disallowed += 1

    print(
        f"[builder] {name}: augmented_policy={policy} — "
        f"added {added_per_split[SPLIT_TRAIN]} to train, "
        f"{added_per_split[SPLIT_VAL]} to val, "
        f"{added_per_split[SPLIT_TEST]} to test; "
        f"dropped {dropped_disallowed} with disallowed-split parent, "
        f"{unknown_group} with unknown group key"
    )
    return {
        "aug_added_to_train": added_per_split[SPLIT_TRAIN],
        "aug_added_to_val": added_per_split[SPLIT_VAL],
        "aug_added_to_test": added_per_split[SPLIT_TEST],
        "aug_dropped_disallowed_split": dropped_disallowed,
        "aug_dropped_unknown_group": unknown_group,
    }


def build_splits(config_path: Path | str, *, force: bool = False) -> Path:
    config_path = Path(config_path)
    cfg, cfg_text = _load_yaml(config_path)
    name = cfg.get("name")
    if not name:
        raise ValueError(f"config {config_path} missing top-level 'name'")
    output_dir = _resolve_cache_dir(cfg.get("output_dir"), "splits")
    split_dir = output_dir / name
    if split_dir.exists() and any(split_dir.iterdir()) and not force:
        raise FileExistsError(
            f"{split_dir} exists and is non-empty; pass force=True to overwrite"
        )
    split_dir.mkdir(parents=True, exist_ok=True)

    _hf_login_if_needed()

    seed = int(cfg.get("seed", 42))
    stratify_by = cfg.get("stratify_by", "emotion")
    speaker_disjoint = bool(cfg.get("speaker_disjoint", True))
    corpora = cfg.get("corpora", {}) or {}

    # 1) Load each configured corpus → NormalizedRow lists
    loaded: dict[str, list[NormalizedRow]] = {}
    load_stats: dict[str, dict[str, int]] = {}
    for ds_name in ALL_DATASETS:
        ds_cfg = corpora.get(ds_name, {}) or {}
        mode = _normalize_mode(ds_cfg.get("mode"))
        if mode == "off":
            continue
        # Normalize cache_dir to absolute path
        if "cache_dir" in ds_cfg:
            ds_cfg = dict(ds_cfg)
            ds_cfg["cache_dir"] = _resolve_cache_dir(ds_cfg.get("cache_dir"), f"data/{ds_name}")
        rows, stats = load_corpus(ds_name, ds_cfg)
        loaded[ds_name] = rows
        load_stats[ds_name] = stats

    # 2) Split each corpus independently
    assignments: dict[str, list[NormalizedRow]] = {s: [] for s in ALL_SPLITS}
    kazemo_resolved: dict[str, Any] | None = None
    builder_stats: dict[str, dict[str, Any]] = {}
    for ds_name, rows in loaded.items():
        ds_cfg = dict(corpora.get(ds_name, {}))
        if "cache_dir" in ds_cfg:
            ds_cfg["cache_dir"] = _resolve_cache_dir(ds_cfg.get("cache_dir"), f"data/{ds_name}")
        idx_map, resolved, bstats = _split_corpus(
            ds_name,
            ds_cfg,
            rows,
            global_stratify_by=stratify_by,
            global_seed=seed,
            speaker_disjoint=speaker_disjoint,
        )
        if resolved and ds_name == DATASET_KAZEMO:
            kazemo_resolved = resolved
        builder_stats[ds_name] = bstats
        for split, idxs in idx_map.items():
            for i in idxs:
                assignments[split].append(rows[i])

    # 3) Sanity checks
    checks = cfg.get("checks", {}) or {}
    fail_spk = bool(checks.get("fail_on_speaker_overlap", True))
    fail_row = bool(checks.get("fail_on_row_overlap", True))
    max_drift = float(checks.get("max_ratio_drift_pp", 5))
    min_emo = int(checks.get("min_samples_per_emotion_per_split", 0))

    errors: list[str] = []
    spk_overlap = _check_speaker_disjoint(assignments) if speaker_disjoint else []
    if spk_overlap:
        msg = f"Speaker leakage (dataset:speaker): {sorted(spk_overlap)[:20]} (total={len(spk_overlap)})"
        (errors if fail_spk else []).append(msg)
        print(f"[builder][warn] {msg}")

    row_overlap = _check_row_id_disjoint(assignments)
    if row_overlap:
        msg = f"(dataset,row_id) overlap across splits: {row_overlap[:10]} (total={len(row_overlap)})"
        (errors if fail_row else []).append(msg)
        print(f"[builder][warn] {msg}")

    if min_emo > 0:
        emo_errs = _check_emotion_coverage(assignments, min_emo)
        if emo_errs:
            print(f"[builder][warn] emotion coverage below threshold: {emo_errs}")

    drift_errs = _check_ratio_drift(assignments, cfg, max_drift)
    if drift_errs:
        msg = f"ratio drift exceeds {max_drift} pp: {drift_errs}"
        print(f"[builder][warn] {msg}")

    # 4) Build per-corpus dropped-rows report (source → load → split → final)
    dropped_rows: dict[str, dict[str, Any]] = {}
    for ds_name in loaded:
        lstat = load_stats.get(ds_name, {})
        bstat = builder_stats.get(ds_name, {})
        assigned = sum(
            1 for s in ALL_SPLITS for r in assignments[s] if r["dataset"] == ds_name
        )
        reasons: dict[str, int] = {}
        if lstat.get("dropped_at_source_aug"):
            reasons["aug_rows_filtered_at_source"] = int(lstat["dropped_at_source_aug"])
        if lstat.get("dropped_missing_speaker"):
            reasons["rows_without_parsable_speaker"] = int(lstat["dropped_missing_speaker"])
        if bstat.get("aug_dropped_val_test_speaker"):
            reasons["aug_copy_of_val_test_speaker"] = int(bstat["aug_dropped_val_test_speaker"])
        if bstat.get("aug_dropped_unknown_speaker"):
            reasons["aug_with_unknown_speaker"] = int(bstat["aug_dropped_unknown_speaker"])
        dropped_rows[ds_name] = {
            "source_rows": int(lstat.get("source_rows", 0)),
            "loaded": int(lstat.get("kept", 0)),
            "assigned": assigned,
            "augmented_policy": bstat.get("augmented_policy"),
            "aug_added_to_train": int(bstat.get("aug_added_to_train", 0)),
            "reasons": reasons,
        }

    # 4b) Aug leak check — derive policy per dataset and verify no aug sibling
    # of a disallowed-split parent slipped through. `speaker_id` on
    # NormalizedRow doubles as the parent-group key (actual speaker for
    # batch01 / source_recording for batch02). Fails loudly if any aug row
    # sits in a split not permitted by its per-dataset policy.
    leak_check = _compute_leak_check(assignments, corpora)
    if not leak_check["passed"]:
        msg = f"aug leak detected: {leak_check}"
        errors.append(msg)
        print(f"[builder][error] {msg}")

    # 5) Write parquet + summary.json
    paths = write_manifests(split_dir, assignments)
    summary = build_summary(
        config=cfg,
        config_text=cfg_text,
        repo_root=REPO_ROOT,
        assignments=assignments,
        kazemo_resolved=kazemo_resolved,
        extra={
            "ratio_drift_errors": drift_errs,
            "emotion_coverage_warnings": _check_emotion_coverage(assignments, min_emo)
            if min_emo > 0
            else [],
            "dropped_rows": dropped_rows,
            "leak_check": leak_check,
        },
    )
    write_summary(split_dir, summary)
    # Also save the resolved config for provenance.
    (split_dir / "config.yaml").write_text(cfg_text)

    print(f"[builder] wrote {paths} + summary.json in {split_dir}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return split_dir
