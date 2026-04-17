"""Parquet + summary.json IO for split manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from splits.schema import (
    ALL_SPLITS,
    MANIFEST_COLUMNS,
    NormalizedRow,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)

BUILDER_VERSION = "1"


def _arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("dataset", pa.string()),
            pa.field("row_id", pa.string()),
            pa.field("source_index", pa.int64()),
            pa.field("speaker_id", pa.string()),
            pa.field("emotion", pa.string()),
            pa.field("gender", pa.string()),
            pa.field("age_category", pa.string()),
            # Nullable bool: stored as int8 (-1/0/1) would be a pain; arrow has nullable bool.
            pa.field("augmented", pa.bool_()),
            pa.field("split", pa.string()),
        ]
    )


def _rows_to_arrow(rows: Iterable[dict[str, Any]]) -> pa.Table:
    # Normalize dicts into column-oriented arrays and let pyarrow handle nulls.
    cols: dict[str, list[Any]] = {c: [] for c in MANIFEST_COLUMNS}
    for r in rows:
        for c in MANIFEST_COLUMNS:
            cols[c].append(r.get(c))
    # Cast row_id/source_index/augmented robustly
    row_ids = [str(v) if v is not None else None for v in cols["row_id"]]
    source_indices = [int(v) if v is not None else None for v in cols["source_index"]]
    augmented = [bool(v) if v is not None else None for v in cols["augmented"]]

    arrays = {
        "dataset": pa.array(cols["dataset"], type=pa.string()),
        "row_id": pa.array(row_ids, type=pa.string()),
        "source_index": pa.array(source_indices, type=pa.int64()),
        "speaker_id": pa.array(cols["speaker_id"], type=pa.string()),
        "emotion": pa.array(cols["emotion"], type=pa.string()),
        "gender": pa.array(cols["gender"], type=pa.string()),
        "age_category": pa.array(cols["age_category"], type=pa.string()),
        "augmented": pa.array(augmented, type=pa.bool_()),
        "split": pa.array(cols["split"], type=pa.string()),
    }
    return pa.table(arrays, schema=_arrow_schema())


def write_manifests(
    split_dir: Path,
    assignments: dict[str, list[NormalizedRow]],
) -> dict[str, Path]:
    """Write one parquet per split. Returns mapping split → path."""
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split_name in ALL_SPLITS:
        rows = [dict(r, split=split_name) for r in assignments.get(split_name, [])]
        table = _rows_to_arrow(rows)
        path = split_dir / f"{split_name}.parquet"
        pq.write_table(table, path, compression="snappy")
        paths[split_name] = path
    return paths


def read_manifest(path: Path) -> list[dict[str, Any]]:
    t = pq.read_table(path)
    d = t.to_pydict()
    n = len(d["dataset"])
    return [{c: d[c][i] for c in d} for i in range(n)]


def read_manifests(split_dir: Path) -> dict[str, list[dict[str, Any]]]:
    split_dir = Path(split_dir)
    out: dict[str, list[dict[str, Any]]] = {}
    for split_name in ALL_SPLITS:
        p = split_dir / f"{split_name}.parquet"
        if p.exists():
            out[split_name] = read_manifest(p)
        else:
            out[split_name] = []
    return out


def _git_sha(cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def config_sha256(config_text: str) -> str:
    return hashlib.sha256(config_text.encode("utf-8")).hexdigest()


def build_summary(
    *,
    config: dict[str, Any],
    config_text: str,
    repo_root: Path,
    assignments: dict[str, list[NormalizedRow]],
    kazemo_resolved: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total = sum(len(v) for v in assignments.values())
    per_split = {s: len(v) for s, v in assignments.items()}

    per_ds_split: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in ALL_SPLITS})
    per_split_emotion: dict[str, Counter] = {s: Counter() for s in ALL_SPLITS}
    per_split_gender: dict[str, Counter] = {s: Counter() for s in ALL_SPLITS}
    per_split_age: dict[str, Counter] = {s: Counter() for s in ALL_SPLITS}
    per_split_speakers: dict[str, set] = {s: set() for s in ALL_SPLITS}
    per_ds_split_speakers: dict[tuple[str, str], set] = defaultdict(set)

    for s, rows in assignments.items():
        for r in rows:
            ds = r["dataset"]
            per_ds_split[ds][s] += 1
            if r.get("emotion"):
                per_split_emotion[s][r["emotion"]] += 1
            if r.get("gender"):
                per_split_gender[s][r["gender"]] += 1
            if r.get("age_category"):
                per_split_age[s][r["age_category"]] += 1
            spk = r.get("speaker_id")
            if spk:
                per_split_speakers[s].add(spk)
                per_ds_split_speakers[(ds, s)].add(spk)

    summary: dict[str, Any] = {
        "builder_version": BUILDER_VERSION,
        "name": config.get("name"),
        "seed": config.get("seed"),
        "stratify_by": config.get("stratify_by"),
        "speaker_disjoint": config.get("speaker_disjoint"),
        "git_sha": _git_sha(repo_root),
        "config_sha256": config_sha256(config_text),
        "totals": {"total": total, **per_split},
        "per_corpus": {
            ds: dict(per_ds_split[ds])
            for ds in sorted(per_ds_split)
        },
        "emotion_counts": {s: dict(sorted(c.items())) for s, c in per_split_emotion.items()},
        "gender_counts": {s: dict(sorted(c.items())) for s, c in per_split_gender.items()},
        "age_counts": {s: dict(sorted(c.items())) for s, c in per_split_age.items()},
        "speaker_counts": {s: len(v) for s, v in per_split_speakers.items()},
        "speakers_per_corpus_split": {
            f"{ds}:{sp}": sorted(v) for (ds, sp), v in per_ds_split_speakers.items()
        },
    }
    if kazemo_resolved:
        summary["kazemo_resolved"] = kazemo_resolved
    if extra:
        summary.update(extra)
    return summary


def write_summary(split_dir: Path, summary: dict[str, Any]) -> Path:
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / "summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return path
