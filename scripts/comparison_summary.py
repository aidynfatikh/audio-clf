#!/usr/bin/env python3
"""Aggregate results/comparison/** into JSON + CSV + Markdown summaries.

Each ``comparison/evaluate.py`` run drops per-corpus ``<corpus>.json`` files
plus a ``summary.json`` under ``results/comparison/<source_tag>/<model_tag>/``.
This script walks that tree, pulls one row per (source, model, corpus) pair,
and emits:

* ``results/comparison/results.json`` — structured, model-keyed view that
  preserves per-class accuracy and class supports for every (corpus, model)
  pair. This is the source of truth for downstream analysis / paper tables.
* ``results/comparison/results.csv`` — flat one-row-per-pair table (no per-
  class breakdown — that lives in JSON).
* ``results/comparison/results.md`` — leaderboard pivot (rows = model,
  columns = corpus, cell = ``acc (n_scored)``), sorted by mean accuracy desc.

Stdlib-only; no pandas dep.

Usage:
    python scripts/comparison_summary.py
    python scripts/comparison_summary.py --root results/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CSV_FIELDS = ("source_tag", "model_tag", "corpus", "accuracy",
              "n_scored", "n_skipped_gt_oov", "n_predicted_unmapped", "n_total")


def _collect_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for corpus_json in sorted(root.glob("*/*/*.json")):
        if corpus_json.name == "summary.json":
            continue
        try:
            data = json.loads(corpus_json.read_text())
        except Exception as e:
            print(f"[summary] skip {corpus_json}: {e}")
            continue
        if "accuracy" not in data or "corpus" not in data:
            continue
        model_dir = corpus_json.parent
        source_dir = model_dir.parent
        rows.append({
            "source_tag": source_dir.name,
            "model_tag": model_dir.name,
            "corpus": data["corpus"],
            "accuracy": float(data.get("accuracy", 0.0)),
            "n_scored": int(data.get("n_scored", 0)),
            "n_skipped_gt_oov": int(data.get("n_skipped_gt_oov", 0)),
            "n_predicted_unmapped": int(data.get("n_predicted_unmapped", 0)),
            "n_total": int(data.get("n_total", 0)),
            "per_class_accuracy": data.get("per_class_accuracy", {}) or {},
            "support": data.get("support", {}) or {},
            "confusion": data.get("confusion", {}) or {},
        })
    return rows


def _write_csv(rows: list[dict], out_path: Path) -> None:
    """Flat one-row-per-pair CSV. Per-class breakdowns live in the JSON."""
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["source_tag"], x["corpus"], -x["accuracy"])):
            w.writerow({k: r[k] for k in CSV_FIELDS})


def _write_json(rows: list[dict], out_path: Path) -> None:
    """Model-keyed structured JSON with per-class accuracy + supports.

    Top-level shape:
        {
          "models": {
            "<model_tag>": {
              "mean_accuracy": <avg over (source, corpus) pairs this model has>,
              "n_results": <count of pairs>,
              "results": {
                "<source_tag>": {
                  "<corpus>": {
                    "accuracy": ...,
                    "n_scored": ..., "n_total": ...,
                    "n_skipped_gt_oov": ..., "n_predicted_unmapped": ...,
                    "per_class_accuracy": {label: float, ...},
                    "support": {label: int, ...},
                    "confusion": {gt_label: {pred_label: int, ...}, ...},
                  }, ...
                }, ...
              }
            }, ...
          }
        }
    """
    by_model: dict[str, dict] = {}
    for r in rows:
        m = r["model_tag"]
        bucket = by_model.setdefault(m, {"mean_accuracy": 0.0, "n_results": 0, "results": {}})
        src_bucket = bucket["results"].setdefault(r["source_tag"], {})
        src_bucket[r["corpus"]] = {
            "accuracy": r["accuracy"],
            "n_scored": r["n_scored"],
            "n_total": r["n_total"],
            "n_skipped_gt_oov": r["n_skipped_gt_oov"],
            "n_predicted_unmapped": r["n_predicted_unmapped"],
            "per_class_accuracy": r["per_class_accuracy"],
            "support": r["support"],
            "confusion": r["confusion"],
        }

    for m, bucket in by_model.items():
        accs = [c["accuracy"]
                for src in bucket["results"].values()
                for c in src.values()]
        bucket["n_results"] = len(accs)
        bucket["mean_accuracy"] = (sum(accs) / len(accs)) if accs else 0.0

    # Sort models in the file by mean_accuracy desc for readability — JSON
    # preserves insertion order in Python 3.7+, and the human consumer of this
    # file (paper tables, plots) gets the leaderboard ordering for free.
    ordered = dict(sorted(
        by_model.items(), key=lambda kv: -kv[1]["mean_accuracy"],
    ))
    out_path.write_text(json.dumps({"models": ordered}, indent=2, ensure_ascii=False))


def _full_coverage_filter(
    rows: list[dict],
) -> tuple[list[tuple[str, str]], list[str], dict, list[tuple[str, str]], list[str]]:
    """Restrict columns and rows so the resulting pivot has zero gaps.

    Strategy:
      1. Build the full set of (source, corpus) columns and model rows present
         on disk.
      2. Iteratively drop the column with the fewest models, then the model
         with the fewest columns, until what remains is a rectangle where
         every (model, column) pair has a result.

    Returns (col_keys, model_keys, cell, dropped_cols, dropped_models).
    """
    cell: dict[tuple[str, str, str], dict] = {
        (r["model_tag"], r["source_tag"], r["corpus"]): r for r in rows
    }
    cols = {(r["source_tag"], r["corpus"]) for r in rows}
    models = {r["model_tag"] for r in rows}
    dropped_cols: list[tuple[str, str]] = []
    dropped_models: list[str] = []

    # Greedy removal: at each step, the (col, model) with the worst coverage
    # ratio gets dropped. Tie-breakers: prefer dropping columns over models
    # (columns are typically the stale ones); within a kind, drop the smaller
    # axis-coverage first.
    while cols and models:
        col_cov = {
            (s, c): sum(1 for m in models if (m, s, c) in cell)
            for (s, c) in cols
        }
        model_cov = {
            m: sum(1 for (s, c) in cols if (m, s, c) in cell)
            for m in models
        }
        if min(col_cov.values()) == len(models) and min(model_cov.values()) == len(cols):
            break  # already a full rectangle
        worst_col_count = min(col_cov.values())
        worst_model_count = min(model_cov.values())
        col_ratio = worst_col_count / max(1, len(models))
        model_ratio = worst_model_count / max(1, len(cols))
        if col_ratio <= model_ratio:
            victim = min(col_cov, key=lambda k: (col_cov[k], k))
            cols.discard(victim)
            dropped_cols.append(victim)
        else:
            victim = min(model_cov, key=lambda k: (model_cov[k], k))
            models.discard(victim)
            dropped_models.append(victim)

    col_keys = sorted(cols)
    model_keys = sorted(models)
    return col_keys, model_keys, cell, dropped_cols, dropped_models


def _write_markdown(rows: list[dict], out_path: Path, include_sparse: bool) -> None:
    if include_sparse:
        cell = {(r["model_tag"], r["source_tag"], r["corpus"]): r for r in rows}
        col_keys = sorted({(r["source_tag"], r["corpus"]) for r in rows})
        model_keys = sorted({r["model_tag"] for r in rows})
        dropped_cols: list[tuple[str, str]] = []
        dropped_models: list[str] = []
    else:
        col_keys, model_keys, cell, dropped_cols, dropped_models = \
            _full_coverage_filter(rows)

    def _mean_acc(model: str) -> float:
        accs = [cell[(model, s, c)]["accuracy"]
                for (s, c) in col_keys
                if (model, s, c) in cell]
        return sum(accs) / len(accs) if accs else 0.0

    def _macro_class_acc(model: str) -> float:
        """Unweighted mean of per-class accuracies across all (corpus, class)
        cells for this model — a quick balanced-accuracy view that ignores
        class frequency. Sources with no per-class breakdown are skipped."""
        vals: list[float] = []
        for (s, c) in col_keys:
            r = cell.get((model, s, c))
            if not r:
                continue
            pca = r.get("per_class_accuracy") or {}
            vals.extend(float(v) for v in pca.values())
        return sum(vals) / len(vals) if vals else 0.0

    model_keys = sorted(model_keys, key=lambda m: -_mean_acc(m))

    header = ["model", "mean_acc", "macro_class_acc"] + [f"{s}/{c}" for (s, c) in col_keys]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for m in model_keys:
        row = [m, f"{_mean_acc(m):.3f}", f"{_macro_class_acc(m):.3f}"]
        for (s, c) in col_keys:
            r = cell.get((m, s, c))
            row.append(f"{r['accuracy']:.3f} (n={r['n_scored']})" if r else "—")
        lines.append("| " + " | ".join(row) + " |")

    preamble = [
        "# Comparison summary",
        "",
        "_Pivot view of overall accuracy on the held-out test set. Per-class "
        "breakdowns are in `results.json` "
        "(`models.<tag>.results.<source>.<corpus>.per_class_accuracy`). "
        "`mean_acc` is the mean of overall accuracies across the listed "
        "(source, corpus) pairs; `macro_class_acc` is the unweighted mean of "
        "per-class accuracies across the same cells (less sensitive to class "
        "imbalance, closer to a balanced-accuracy reading)._",
        "",
    ]

    if dropped_cols or dropped_models:
        preamble += [
            "## Coverage filter applied",
            "",
            "_Columns / models without full coverage were dropped so the "
            "table is gap-free. To see the raw sparse pivot pass "
            "`--include-sparse`._",
            "",
        ]
        if dropped_cols:
            preamble.append("**Dropped columns (incomplete coverage):**\n")
            for s, c in sorted(dropped_cols):
                preamble.append(f"- `{s}/{c}`")
            preamble.append("")
        if dropped_models:
            preamble.append("**Dropped models (incomplete coverage):**\n")
            for m in sorted(dropped_models):
                preamble.append(f"- `{m}`")
            preamble.append("")

    out_path.write_text("\n".join(preamble) + "\n".join(lines) + "\n")


def _write_test_per_class(rows: list[dict], out_path: Path) -> None:
    """Long-form per-class accuracy table — one row per (model, source, corpus,
    class). Useful for the paper's per-class breakdown section."""
    fields = ("source_tag", "model_tag", "corpus", "class",
              "accuracy", "support", "overall_accuracy", "n_scored")
    long_rows: list[dict] = []
    for r in rows:
        pca = r.get("per_class_accuracy") or {}
        sup = r.get("support") or {}
        for cls in sorted(set(pca) | set(sup)):
            long_rows.append({
                "source_tag": r["source_tag"],
                "model_tag": r["model_tag"],
                "corpus": r["corpus"],
                "class": cls,
                "accuracy": float(pca.get(cls, 0.0)),
                "support": int(sup.get(cls, 0)),
                "overall_accuracy": float(r["accuracy"]),
                "n_scored": int(r["n_scored"]),
            })
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(long_rows, key=lambda x: (
            x["source_tag"], x["corpus"], x["model_tag"], x["class"]
        )):
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="results/comparison",
                    help="Root directory to scan (default results/comparison)")
    ap.add_argument(
        "--include-sparse", action="store_true",
        help="Keep all (model, source/corpus) pairs in the Markdown pivot, "
             "even if some cells are missing. Default behaviour drops "
             "incomplete columns/models so the table is gap-free.",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"No results dir: {root}")

    rows = _collect_rows(root)
    if not rows:
        raise SystemExit(f"No per-corpus JSON files found under {root}")

    json_path = root / "results.json"
    csv_path = root / "results.csv"
    md_path = root / "results.md"
    per_class_path = root / "results_per_class.csv"
    _write_json(rows, json_path)
    _write_csv(rows, csv_path)
    _write_markdown(rows, md_path, include_sparse=args.include_sparse)
    _write_test_per_class(rows, per_class_path)
    print(f"[summary] wrote {json_path} ({len(rows)} rows, "
          f"{len({r['model_tag'] for r in rows})} models)")
    print(f"[summary] wrote {csv_path}")
    print(f"[summary] wrote {md_path} "
          f"({'sparse' if args.include_sparse else 'gap-free'})")
    print(f"[summary] wrote {per_class_path}")


if __name__ == "__main__":
    main()
