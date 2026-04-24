#!/usr/bin/env python3
"""Aggregate results/comparison/** into a single CSV + Markdown summary.

Each ``comparison/evaluate.py`` run drops per-corpus ``<corpus>.json`` files
plus a ``summary.json`` under ``results/comparison/<source_tag>/<model_tag>/``.
This script walks that tree, pulls one row per (source, model, corpus) pair,
and emits:

* ``results/comparison/summary.csv`` — flat table, one row per pair.
* ``results/comparison/summary.md``  — leaderboard pivot (rows = model,
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


FIELDS = ("source_tag", "model_tag", "corpus", "accuracy",
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
            "accuracy": data.get("accuracy", 0.0),
            "n_scored": data.get("n_scored", 0),
            "n_skipped_gt_oov": data.get("n_skipped_gt_oov", 0),
            "n_predicted_unmapped": data.get("n_predicted_unmapped", 0),
            "n_total": data.get("n_total", 0),
        })
    return rows


def _write_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["source_tag"], x["corpus"], -x["accuracy"])):
            w.writerow({k: r[k] for k in FIELDS})


def _write_markdown(rows: list[dict], out_path: Path) -> None:
    # Pivot: rows = model_tag, columns = "<source>/<corpus>"
    col_keys: list[tuple[str, str]] = sorted({(r["source_tag"], r["corpus"]) for r in rows})
    model_keys: list[str] = sorted({r["model_tag"] for r in rows})

    cell: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        cell[(r["model_tag"], r["source_tag"], r["corpus"])] = r

    # Sort models by mean accuracy across cells they have.
    def _mean_acc(model: str) -> float:
        accs = [cell[(model, s, c)]["accuracy"]
                for (s, c) in col_keys
                if (model, s, c) in cell]
        return sum(accs) / len(accs) if accs else 0.0

    model_keys.sort(key=_mean_acc, reverse=True)

    header = ["model", "mean_acc"] + [f"{s}/{c}" for (s, c) in col_keys]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for m in model_keys:
        row = [m, f"{_mean_acc(m):.3f}"]
        for (s, c) in col_keys:
            r = cell.get((m, s, c))
            row.append(f"{r['accuracy']:.3f} (n={r['n_scored']})" if r else "—")
        lines.append("| " + " | ".join(row) + " |")

    out_path.write_text("# Comparison summary\n\n" + "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="results/comparison",
                    help="Root directory to scan (default results/comparison)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"No results dir: {root}")

    rows = _collect_rows(root)
    if not rows:
        raise SystemExit(f"No per-corpus JSON files found under {root}")

    csv_path = root / "summary.csv"
    md_path = root / "summary.md"
    _write_csv(rows, csv_path)
    _write_markdown(rows, md_path)
    print(f"[summary] wrote {csv_path} ({len(rows)} rows)")
    print(f"[summary] wrote {md_path}")


if __name__ == "__main__":
    main()
