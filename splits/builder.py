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
    DATASET_BATCH01,
    DATASET_BATCH02,
    DATASET_KAZEMO,
    NormalizedRow,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from splits.sources import load_corpus
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
    """Return list of speakers that appear in more than one split."""
    spk_to_splits: dict[str, set[str]] = {}
    for split, rows in assignments.items():
        for r in rows:
            spk = r.get("speaker_id")
            if not spk:
                continue
            spk_to_splits.setdefault(spk, set()).add(split)
    return [spk for spk, splits in spk_to_splits.items() if len(splits) > 1]


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
) -> tuple[dict[str, list[int]], dict[str, Any] | None]:
    """Return (indices_per_split, kazemo_resolved_if_any)."""
    mode = _normalize_mode(ds_cfg.get("mode"))
    if mode == "off":
        return {SPLIT_TRAIN: [], SPLIT_VAL: [], SPLIT_TEST: []}, None
    if mode == "train_only":
        return _assign_train_only(rows), None
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
            valtest_speaker=valtest_spk,
            valtest_ratio=valtest_ratio,
            seed=ds_cfg.get("seed", global_seed),
        )
        resolved = {
            "discovered_speakers": all_speakers,
            "train_speakers": sorted(train_spk),
            "valtest_speaker": valtest_spk,
        }
        return idx, resolved

    # batch01/batch02 — stratified grouped
    stratify_by = ds_cfg.get("stratify_by", global_stratify_by)
    group_field = "speaker_id" if speaker_disjoint else "row_id"
    ratios = ds_cfg.get("ratios", {SPLIT_TRAIN: 0.7, SPLIT_VAL: 0.15, SPLIT_TEST: 0.15})
    idx = stratified_grouped_three_way(
        rows,
        ratios=ratios,
        stratify_by=stratify_by,
        group_by_field=group_field,
        seed=ds_cfg.get("seed", global_seed),
    )
    return idx, None


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
    for ds_name in ALL_DATASETS:
        ds_cfg = corpora.get(ds_name, {}) or {}
        mode = _normalize_mode(ds_cfg.get("mode"))
        if mode == "off":
            continue
        # Normalize cache_dir to absolute path
        if "cache_dir" in ds_cfg:
            ds_cfg = dict(ds_cfg)
            ds_cfg["cache_dir"] = _resolve_cache_dir(ds_cfg.get("cache_dir"), f"data/{ds_name}")
        loaded[ds_name] = load_corpus(ds_name, ds_cfg)

    # 2) Split each corpus independently
    assignments: dict[str, list[NormalizedRow]] = {s: [] for s in ALL_SPLITS}
    kazemo_resolved: dict[str, Any] | None = None
    for ds_name, rows in loaded.items():
        ds_cfg = dict(corpora.get(ds_name, {}))
        if "cache_dir" in ds_cfg:
            ds_cfg["cache_dir"] = _resolve_cache_dir(ds_cfg.get("cache_dir"), f"data/{ds_name}")
        idx_map, resolved = _split_corpus(
            ds_name,
            ds_cfg,
            rows,
            global_stratify_by=stratify_by,
            global_seed=seed,
            speaker_disjoint=speaker_disjoint,
        )
        if resolved and ds_name == DATASET_KAZEMO:
            kazemo_resolved = resolved
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
        msg = f"Speakers appearing in multiple splits: {sorted(spk_overlap)[:20]} (total={len(spk_overlap)})"
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

    # 4) Write parquet + summary.json
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
        },
    )
    write_summary(split_dir, summary)
    # Also save the resolved config for provenance.
    (split_dir / "config.yaml").write_text(cfg_text)

    print(f"[builder] wrote {paths} + summary.json in {split_dir}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return split_dir
