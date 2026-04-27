#!/usr/bin/env python
"""Upload a materialized split to the Hugging Face Hub as a DatasetDict.

Includes per-row provenance (source_dataset, source_row_id, source_index,
speaker_id, augmented) and a generated dataset card with tables + matplotlib
distribution plots, built from summary.json + config.yaml. Also uploads the
original config.yaml and summary.json as auditable artifacts.

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
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi, login as hf_login

from splits.materialize import load_split_as_hf_dataset


SPLITS = ("train", "val", "test")
SPLIT_COLORS = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}
EMOTIONS = ("angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised")


def _fmt_int(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def _counts_table(
    counts_by_split: dict[str, dict[str, int]],
    label_col: str = "label",
) -> str:
    keys = sorted({k for d in counts_by_split.values() for k in (d or {}).keys()})
    if not keys:
        return "_(none)_"
    cols = [s for s in SPLITS if s in counts_by_split] + [
        s for s in counts_by_split if s not in SPLITS
    ]
    header = f"| {label_col} | " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    lines = [header, sep]
    for k in keys:
        row = [k] + [_fmt_int((counts_by_split.get(s) or {}).get(k, 0)) for s in cols]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig_split_totals(summary: dict[str, Any], out_dir: Path) -> Path:
    totals = summary.get("totals", {}) or {}
    aug = (summary.get("leak_check") or {}).get("per_split_aug_count", {}) or {}
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    xs = list(SPLITS)
    aug_vals = [int(aug.get(s, 0)) for s in xs]
    orig_vals = [int(totals.get(s, 0)) - a for s, a in zip(xs, aug_vals)]
    ax.bar(xs, orig_vals, label="original", color="#4C78A8")
    ax.bar(xs, aug_vals, bottom=orig_vals, label="augmented", color="#F58518")
    for i, s in enumerate(xs):
        total = orig_vals[i] + aug_vals[i]
        ax.text(i, total, f"{total:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("rows")
    ax.set_title("Rows per split (original + augmented)")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    return _save(fig, out_dir, "split_totals.png")


def _fig_per_corpus(summary: dict[str, Any], out_dir: Path) -> Path:
    per_corpus = summary.get("per_corpus", {}) or {}
    corpora = sorted(per_corpus.keys())
    if not corpora:
        return None
    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(corpora)), 3.6))
    width = 0.26
    xs = range(len(corpora))
    for i, s in enumerate(SPLITS):
        vals = [int((per_corpus.get(c) or {}).get(s, 0)) for c in corpora]
        offsets = [x + (i - 1) * width for x in xs]
        ax.bar(offsets, vals, width=width, label=s, color=SPLIT_COLORS[s])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(corpora)
    ax.set_ylabel("rows")
    ax.set_title("Rows per corpus × split")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    return _save(fig, out_dir, "per_corpus.png")


def _fig_emotion(summary: dict[str, Any], out_dir: Path) -> Path:
    emo = summary.get("emotion_counts", {}) or {}
    labels = sorted({k for s in SPLITS for k in (emo.get(s) or {}).keys()})
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(labels)), 3.8))
    width = 0.26
    xs = range(len(labels))
    for i, s in enumerate(SPLITS):
        vals = [int((emo.get(s) or {}).get(k, 0)) for k in labels]
        offsets = [x + (i - 1) * width for x in xs]
        ax.bar(offsets, vals, width=width, label=s, color=SPLIT_COLORS[s])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("rows")
    ax.set_title("Emotion distribution per split")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    return _save(fig, out_dir, "emotion_distribution.png")


def _fig_emotion_train_aug(summary: dict[str, Any], out_dir: Path) -> Path:
    """For each corpus, train-split per-emotion original vs augmented (stacked)."""
    pce = summary.get("per_corpus_emotion_counts", {}) or {}
    corpora = [c for c in sorted(pce.keys()) if pce[c].get("train")]
    if not corpora:
        return None
    n = len(corpora)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, c in zip(axes, corpora):
        per_emo = pce[c].get("train", {}) or {}
        labels = sorted(per_emo.keys())
        orig = [int(per_emo[k]["total"]) - int(per_emo[k].get("augmented", 0)) for k in labels]
        aug = [int(per_emo[k].get("augmented", 0)) for k in labels]
        ax.bar(labels, orig, color="#4C78A8", label="original")
        ax.bar(labels, aug, bottom=orig, color="#F58518", label="augmented")
        ax.set_title(f"{c} (train)")
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment("right")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("rows")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Train-split per-emotion: original + augmented (per corpus)", y=1.02)
    return _save(fig, out_dir, "train_emotion_aug_per_corpus.png")


def _fig_simple_dist(
    counts_by_split: dict[str, dict[str, int]],
    title: str,
    out_dir: Path,
    name: str,
) -> Path:
    labels = sorted({k for s in SPLITS for k in (counts_by_split.get(s) or {}).keys()})
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(max(5.5, 0.8 * len(labels) + 2.0), 3.4))
    width = 0.26
    xs = range(len(labels))
    for i, s in enumerate(SPLITS):
        vals = [int((counts_by_split.get(s) or {}).get(k, 0)) for k in labels]
        offsets = [x + (i - 1) * width for x in xs]
        ax.bar(offsets, vals, width=width, label=s, color=SPLIT_COLORS[s])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylabel("rows")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    return _save(fig, out_dir, name)


def _generate_figures(summary: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs: dict[str, Path] = {}
    if (p := _fig_split_totals(summary, out_dir)):
        figs["split_totals"] = p
    if (p := _fig_per_corpus(summary, out_dir)):
        figs["per_corpus"] = p
    if (p := _fig_emotion(summary, out_dir)):
        figs["emotion"] = p
    if (p := _fig_emotion_train_aug(summary, out_dir)):
        figs["emotion_train_aug"] = p
    if (p := _fig_simple_dist(
        summary.get("gender_counts", {}) or {}, "Gender per split", out_dir, "gender.png"
    )):
        figs["gender"] = p
    if (p := _fig_simple_dist(
        summary.get("age_counts", {}) or {}, "Age category per split", out_dir, "age.png"
    )):
        figs["age"] = p
    return figs


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------
def _per_corpus_aug_table(summary: dict[str, Any]) -> str:
    per_corpus = summary.get("per_corpus", {}) or {}
    aug = (summary.get("leak_check") or {}).get("per_corpus_aug_count", {}) or {}
    if not per_corpus:
        return "_(none)_"
    lines = [
        "| corpus | split | original | augmented | total |",
        "|---|---|---:|---:|---:|",
    ]
    for c in sorted(per_corpus.keys()):
        for s in SPLITS:
            total = int((per_corpus.get(c) or {}).get(s, 0))
            a = int((aug.get(c) or {}).get(s, 0))
            o = total - a
            if total == 0 and a == 0:
                continue
            lines.append(f"| `{c}` | {s} | {_fmt_int(o)} | {_fmt_int(a)} | {_fmt_int(total)} |")
    return "\n".join(lines)


def _per_corpus_emotion_aug_tables(summary: dict[str, Any]) -> str:
    pce = summary.get("per_corpus_emotion_counts", {}) or {}
    if not pce:
        return "_(none)_"
    blocks: list[str] = []
    for c in sorted(pce.keys()):
        per_split = pce[c]
        # Skip if entire corpus has zero augmented rows anywhere.
        if not any(
            int(d.get("augmented", 0)) > 0
            for s in SPLITS
            for d in (per_split.get(s) or {}).values()
        ):
            # Still show emotion totals per split — useful even without aug.
            pass
        emos = sorted({e for s in SPLITS for e in (per_split.get(s) or {}).keys()})
        if not emos:
            continue
        block = [f"\n**`{c}`** — original / augmented per split\n"]
        block.append("| emotion | " + " | ".join(
            f"{s} (orig+aug)" for s in SPLITS
        ) + " |")
        block.append("|---" + "|---:" * len(SPLITS) + "|")
        for e in emos:
            cells = [e]
            for s in SPLITS:
                d = (per_split.get(s) or {}).get(e) or {"total": 0, "augmented": 0}
                t = int(d.get("total", 0))
                a = int(d.get("augmented", 0))
                o = t - a
                if t == 0:
                    cells.append("—")
                elif a == 0:
                    cells.append(f"{o:,}")
                else:
                    cells.append(f"{o:,} + {a:,}")
            block.append("| " + " | ".join(cells) + " |")
        blocks.append("\n".join(block))
    return "\n".join(blocks) if blocks else "_(none)_"


def _img(figures: dict[str, Path] | None, key: str, alt: str) -> str:
    if not figures or key not in figures:
        return ""
    return f"![{alt}](figures/{figures[key].name})\n"


def _build_card(
    summary: dict[str, Any],
    config: dict[str, Any],
    repo_id: str,
    figures: dict[str, Path] | None = None,
) -> str:
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

    aug_total = sum(int(v) for v in (leak.get("per_split_aug_count") or {}).values())
    orig_total = int(totals.get("total", 0)) - aug_total

    yaml_front = (
        "---\n"
        "task_categories:\n- audio-classification\n"
        "language:\n- kk\n"
        "tags:\n- emotion\n- speech\n- kazakh\n"
        f"pretty_name: {name}\n"
        "size_categories:\n- 10K<n<1M\n"
        "---\n\n"
    )

    p: list[str] = [yaml_front, f"# {name}\n"]
    p.append(
        "Train / val / test splits for Kazakh speech-emotion classification, materialized "
        "from a deterministic config. Splits are speaker-disjoint and stratified by emotion. "
        "Every row carries provenance back to its source corpus and row id.\n"
    )

    # Headline numbers
    p.append("## At a glance\n")
    p.append(
        f"- **Total rows:** {_fmt_int(totals.get('total', 0))} "
        f"(train **{_fmt_int(totals.get('train', 0))}**, "
        f"val **{_fmt_int(totals.get('val', 0))}**, "
        f"test **{_fmt_int(totals.get('test', 0))}**)"
    )
    p.append(
        f"- **Original / augmented:** {_fmt_int(orig_total)} original + "
        f"{_fmt_int(aug_total)} augmented"
    )
    p.append(f"- **Speakers:** train {_fmt_int(spk.get('train', 0))} · "
             f"val {_fmt_int(spk.get('val', 0))} · test {_fmt_int(spk.get('test', 0))}")
    p.append(f"- **Emotions:** {', '.join(EMOTIONS)}")
    p.append("")
    p.append(_img(figures, "split_totals", "Rows per split"))

    # Source corpora
    p.append("## Source corpora\n")
    corpora = config.get("corpora", {}) or {}
    p.append("| corpus | mode | hf_id | aug policy | ratios |")
    p.append("|---|---|---|---|---|")
    for ds, ds_cfg in corpora.items():
        ds_cfg = ds_cfg or {}
        mode = str(ds_cfg.get("mode"))
        hf_id = ds_cfg.get("hf_id", "_(local)_")
        pol = ds_cfg.get("augmented_policy", "—")
        ratios = ds_cfg.get("ratios")
        ratios_s = ", ".join(f"{k}={v}" for k, v in ratios.items()) if ratios else "—"
        p.append(f"| `{ds}` | `{mode}` | `{hf_id}` | `{pol}` | {ratios_s} |")
    p.append("")

    # Per-corpus composition
    p.append("## Per-corpus composition\n")
    p.append(_img(figures, "per_corpus", "Per corpus × split"))
    p.append(_per_corpus_aug_table(summary))
    p.append("")

    # Emotion
    p.append("## Emotion distribution\n")
    p.append(_img(figures, "emotion", "Emotion per split"))
    p.append(_counts_table(emo, label_col="emotion"))
    p.append("")

    # Per-corpus per-emotion w/ augmented breakdown
    p.append("### Per-corpus emotion counts (original + augmented)\n")
    p.append(_img(figures, "emotion_train_aug", "Train-split per-emotion aug per corpus"))
    p.append(_per_corpus_emotion_aug_tables(summary))
    p.append("")

    # Gender / age
    p.append("## Gender\n")
    p.append(_img(figures, "gender", "Gender per split"))
    p.append(_counts_table(gen, label_col="gender"))
    p.append("")

    p.append("## Age category\n")
    p.append(_img(figures, "age", "Age per split"))
    p.append(_counts_table(age, label_col="age"))
    p.append("")

    # Speakers
    p.append("## Speakers\n")
    p.append("| split | unique speakers |")
    p.append("|---|---:|")
    for s in SPLITS:
        p.append(f"| {s} | {_fmt_int(spk.get(s, 0))} |")
    if spk_per_cs:
        p.append("")
        p.append("**Per (corpus, split):**\n")
        p.append("| corpus | train | val | test |")
        p.append("|---|---:|---:|---:|")
        # Build dict {corpus: {split: count}}
        cs: dict[str, dict[str, int]] = {}
        for k, v in spk_per_cs.items():
            ds, _, sp = k.partition(":")
            cs.setdefault(ds, {})[sp] = int(v)
        for ds in sorted(cs):
            row = [f"`{ds}`"] + [_fmt_int(cs[ds].get(s, 0)) for s in SPLITS]
            p.append("| " + " | ".join(row) + " |")
    p.append("")

    # Drops
    if dropped:
        p.append("## Drop accounting (per corpus)\n")
        p.append("| corpus | source rows | loaded | assigned | aug policy | aug → train | drop reasons |")
        p.append("|---|---:|---:|---:|---|---:|---|")
        for ds, info in dropped.items():
            info = info or {}
            reasons = info.get("reasons", {}) or {}
            reasons_s = "; ".join(f"{k}={v}" for k, v in reasons.items()) or "—"
            p.append(
                f"| `{ds}` | {_fmt_int(info.get('source_rows', 0))} | "
                f"{_fmt_int(info.get('loaded', 0))} | {_fmt_int(info.get('assigned', 0))} | "
                f"`{info.get('augmented_policy', '—')}` | "
                f"{_fmt_int(info.get('aug_added_to_train', 0))} | {reasons_s} |"
            )
        p.append("")

    # Leak / integrity
    p.append("## Integrity\n")
    p.append(f"- Speaker-disjoint splits: `{summary.get('speaker_disjoint')}`")
    p.append(f"- Stratify by: `{summary.get('stratify_by')}`")
    p.append(f"- Augmentation leak check: **{'passed' if leak.get('passed') else 'FAILED'}** "
             f"({leak.get('total_violations', 0)} violations)")
    pol = leak.get("policy_per_corpus", {}) or {}
    if pol:
        pol_s = "; ".join(
            f"`{ds}`→{splits if splits else '∅'}" for ds, splits in pol.items()
        )
        p.append(f"- Aug allowed splits per corpus: {pol_s}")
    p.append("")

    # Schema
    p.append("## Schema\n")
    p.append("Each row contains:")
    p.append("- `audio` — Audio (decode=False; raw bytes + sampling_rate)")
    p.append(f"- `emotion` — one of: {', '.join(EMOTIONS)} (or null)")
    p.append("- `gender` — `M` / `F` / null")
    p.append("- `age_category` — child / young / adult / senior / null")
    p.append("- `source_dataset` — origin corpus key (e.g. `batch01`, `batch02`, `kazemo`, `kazattsd`)")
    p.append("- `source_row_id` — stable id within the source corpus")
    p.append("- `source_index` — row index in the source HF split")
    p.append("- `speaker_id` — speaker key (when known)")
    p.append("- `augmented` — true if this row is a synthetic / augmented copy")
    p.append("")

    # Provenance
    p.append("## Build provenance\n")
    p.append(f"- Builder seed: `{seed}`")
    if git_sha:
        p.append(f"- Repo commit: `{git_sha}`")
    if cfg_sha:
        p.append(f"- Config sha256: `{cfg_sha}`")
    p.append("- Auditable artifacts under `build/`: `config.yaml`, `summary.json`")
    p.append("")
    return "\n".join(p)


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
                    help="Skip generating/uploading README.md and figures")
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
        with tempfile.TemporaryDirectory() as td:
            fig_dir = Path(td) / "figures"
            figures = _generate_figures(summary, fig_dir)
            for key, path in figures.items():
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=f"figures/{path.name}",
                    repo_id=args.repo_id,
                    repo_type="dataset",
                )
                print(f"[push] uploaded figure: figures/{path.name}")
            card = _build_card(summary, config, args.repo_id, figures=figures)
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
