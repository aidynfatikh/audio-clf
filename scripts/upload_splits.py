#!/usr/bin/env python
"""Upload a materialized split to the Hugging Face Hub as a DatasetDict.

Includes per-row provenance (source_dataset, source_row_id, source_index,
speaker_id, augmented) and a generated README.md / dataset card built from
summary.json + config.yaml. Also uploads the original config.yaml and
summary.json as auditable artifacts.

Usage:
  python scripts/upload_splits.py \
      --split-dir splits/b1_b2_noaug_v1 \
      --repo-id 01gumano1d/audio-clf-b1b2-noaug-v1 \
      --private                      # or --public

Required env (in .env or shell):
  HF_TOKEN   — Hugging Face access token with write scope on the target repo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi, login as hf_login

from splits.materialize import load_split_as_hf_dataset


def _fmt_int(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def _counts_table(counts_by_split: dict[str, dict[str, int]]) -> str:
    keys = sorted({k for d in counts_by_split.values() for k in (d or {}).keys()})
    if not keys:
        return "_(none)_"
    header = "| label | " + " | ".join(counts_by_split.keys()) + " |"
    sep = "|" + "|".join(["---"] * (len(counts_by_split) + 1)) + "|"
    lines = [header, sep]
    for k in keys:
        row = [k] + [_fmt_int((counts_by_split.get(s) or {}).get(k, 0)) for s in counts_by_split.keys()]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _build_card(summary: dict[str, Any], config: dict[str, Any], repo_id: str) -> str:
    name = summary.get("name") or config.get("name") or repo_id.split("/")[-1]
    seed = summary.get("seed", config.get("seed"))
    totals = summary.get("totals", {}) or {}
    per_corpus = summary.get("per_corpus", {}) or {}
    emo = summary.get("emotion_counts", {}) or {}
    gen = summary.get("gender_counts", {}) or {}
    age = summary.get("age_counts", {}) or {}
    spk = summary.get("speaker_counts", {}) or {}
    spk_per_cs = summary.get("speaker_counts_per_corpus_split", {}) or {}
    dropped = summary.get("dropped_rows", {}) or {}
    leak = summary.get("leak_check", {}) or {}
    git_sha = summary.get("git_sha")
    cfg_sha = summary.get("config_sha256")

    yaml_front = (
        "---\n"
        "task_categories:\n- audio-classification\n"
        "language:\n- kk\n"
        "tags:\n- emotion\n- speech\n- kazakh\n"
        f"pretty_name: {name}\n"
        "---\n\n"
    )

    parts = [yaml_front, f"# {name}\n"]
    parts.append(
        "Materialized train/val/test splits for Kazakh speech-emotion classification, "
        "built from a deterministic YAML config. Each row carries per-row provenance "
        "back to its original source corpus and row id.\n"
    )

    parts.append("## Build provenance\n")
    parts.append(f"- Builder seed: `{seed}`")
    parts.append(f"- Speaker disjoint: `{summary.get('speaker_disjoint')}`")
    parts.append(f"- Stratify by: `{summary.get('stratify_by')}`")
    if git_sha:
        parts.append(f"- Repo commit: `{git_sha}`")
    if cfg_sha:
        parts.append(f"- Config sha256: `{cfg_sha}`")
    parts.append("")

    parts.append("## Source corpora\n")
    corpora = config.get("corpora", {}) or {}
    parts.append("| corpus | mode | hf_id | aug policy | ratios |")
    parts.append("|---|---|---|---|---|")
    for ds, ds_cfg in corpora.items():
        ds_cfg = ds_cfg or {}
        mode = str(ds_cfg.get("mode"))
        hf_id = ds_cfg.get("hf_id", "_(local)_")
        pol = ds_cfg.get("augmented_policy", "—")
        ratios = ds_cfg.get("ratios")
        ratios_s = ", ".join(f"{k}={v}" for k, v in ratios.items()) if ratios else "—"
        parts.append(f"| `{ds}` | `{mode}` | `{hf_id}` | `{pol}` | {ratios_s} |")
    parts.append("")

    parts.append("## Composition\n")
    parts.append(
        f"Total rows: **{_fmt_int(totals.get('total', 0))}** "
        f"(train={_fmt_int(totals.get('train', 0))}, "
        f"val={_fmt_int(totals.get('val', 0))}, "
        f"test={_fmt_int(totals.get('test', 0))})\n"
    )

    parts.append("### Per-corpus row counts\n")
    parts.append(_counts_table(per_corpus))
    parts.append("")

    parts.append("### Emotion distribution\n")
    parts.append(_counts_table(emo))
    parts.append("")
    parts.append("### Gender distribution\n")
    parts.append(_counts_table(gen))
    parts.append("")
    parts.append("### Age distribution\n")
    parts.append(_counts_table(age))
    parts.append("")

    parts.append("### Speaker counts\n")
    parts.append("| split | unique speakers |")
    parts.append("|---|---|")
    for s in ("train", "val", "test"):
        parts.append(f"| {s} | {_fmt_int(spk.get(s, 0))} |")
    if spk_per_cs:
        parts.append("\n_per (corpus, split):_\n")
        parts.append("```")
        for k in sorted(spk_per_cs.keys()):
            parts.append(f"{k}: {spk_per_cs[k]}")
        parts.append("```")
    parts.append("")

    if dropped:
        parts.append("## Dropped rows (per corpus)\n")
        parts.append("| corpus | source_rows | loaded | assigned | aug policy | aug→train | reasons |")
        parts.append("|---|---|---|---|---|---|---|")
        for ds, info in dropped.items():
            info = info or {}
            reasons = info.get("reasons", {}) or {}
            reasons_s = "; ".join(f"{k}={v}" for k, v in reasons.items()) or "—"
            parts.append(
                f"| `{ds}` | {_fmt_int(info.get('source_rows', 0))} | "
                f"{_fmt_int(info.get('loaded', 0))} | {_fmt_int(info.get('assigned', 0))} | "
                f"`{info.get('augmented_policy', '—')}` | "
                f"{_fmt_int(info.get('aug_added_to_train', 0))} | {reasons_s} |"
            )
        parts.append("")

    parts.append("## Leak check\n")
    parts.append(f"- Passed: **{leak.get('passed')}**")
    parts.append(f"- Total violations: {leak.get('total_violations', 0)}")
    parts.append(f"- Aug counts per split: `{leak.get('per_split_aug_count', {})}`")
    parts.append("")

    parts.append("## Schema\n")
    parts.append("Each row contains:")
    parts.append("- `audio` — Audio (decode=False; raw bytes + sampling_rate)")
    parts.append("- `emotion` — one of: angry, disgusted, fearful, happy, neutral, sad, surprised (or null)")
    parts.append("- `gender` — `M` / `F` / null")
    parts.append("- `age_category` — child / young / adult / senior / null")
    parts.append("- `source_dataset` — origin corpus key (e.g. `batch01`, `batch02`, `kazemo`, `kazattsd`)")
    parts.append("- `source_row_id` — stable id within the source corpus")
    parts.append("- `source_index` — row index in the source HF split")
    parts.append("- `speaker_id` — speaker key (when known)")
    parts.append("- `augmented` — true if this row is a synthetic/augmented copy")
    parts.append("")

    parts.append("## Reproducibility\n")
    parts.append(
        "The original `config.yaml` and `summary.json` used to build this dataset are "
        "uploaded under `build/` alongside the parquet shards. Rebuild locally with:\n"
    )
    parts.append("```\npython scripts/build_splits.py --config configs/splits/<name>.yaml\n```")
    parts.append("")
    parts.append(
        "_Note: source corpora are not pinned to a Hub revision in the build config — "
        "if upstream sources change, rebuilding may produce different splits even with "
        "the same seed._\n"
    )
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True, type=Path)
    ap.add_argument("--repo-id", required=True, help="HF dataset repo id, e.g. org/name")
    vis = ap.add_mutually_exclusive_group()
    vis.add_argument("--private", dest="private", action="store_true",
                     help="Create the repo as private (default)")
    vis.add_argument("--public", dest="private", action="store_false",
                     help="Create the repo as public")
    ap.set_defaults(private=True)
    ap.add_argument("--no-provenance", action="store_true",
                    help="Skip per-row provenance columns (smaller schema, less traceable)")
    ap.add_argument("--no-card", action="store_true",
                    help="Skip generating/uploading README.md")
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("HF_TOKEN not set (.env or shell)", file=sys.stderr)
        return 2
    hf_login(token=tok)

    split_dir = args.split_dir.resolve()
    summary_path = split_dir / "summary.json"
    config_path = split_dir / "config.yaml"
    if not summary_path.exists() or not config_path.exists():
        print(f"[push] missing summary.json or config.yaml in {split_dir}", file=sys.stderr)
        return 2

    summary = json.loads(summary_path.read_text())
    config = yaml.safe_load(config_path.read_text()) or {}

    dd = load_split_as_hf_dataset(split_dir, with_provenance=not args.no_provenance)
    print(f"[push] built DatasetDict (provenance={'off' if args.no_provenance else 'on'}):")
    for k, v in dd.items():
        print(f"  {k}: {len(v)} rows; cols={v.column_names}")

    visibility = "private" if args.private else "public"
    print(f"[push] pushing to {args.repo_id} as {visibility}")
    dd.push_to_hub(args.repo_id, private=args.private)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(config_path),
        path_in_repo="build/config.yaml",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(summary_path),
        path_in_repo="build/summary.json",
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    if not args.no_card:
        card = _build_card(summary, config, args.repo_id)
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print("[push] uploaded README.md (dataset card)")

    print(f"[push] done → https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
