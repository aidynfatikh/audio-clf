#!/usr/bin/env python3
"""W&B analysis — fair within-cohort + guarded cross-cohort comparisons.

Reads `results/wandb_export/*.json` → writes `results/wandb_analysis/results.md`.

Design
------
Runs are grouped by **cohort = (HEAD commit at run start, split-variant)**.
Split-variant captures how the val set was constructed for that run, since a
single commit can host multiple experiments (exp1/2 = aug splits, exp3 = noaug
splits, both on commit `e69a27a`). Within a cohort, val-data definition is
identical → metrics are directly comparable.

Exclusions: runs on commit `38b23c8` (04-17) are dropped. Their split config
set `speaker_column: speaker_id` on batch02, which has no such column → splits
silently fell back to per-row, leaking augmented siblings of the same source
recording between train and val.

Fair cross-cohort leaderboards are limited to (corpus, task) pairs where val
definition is provably identical across the listed cohorts — see
`FAIR_ACROSS` below for the exact claim.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = REPO_ROOT / "results" / "wandb_export"
OUT_DIR = REPO_ROOT / "results" / "wandb_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED_COMMITS = {"38b23c8"}

TASKS = ("emotion", "gender", "age")
CORPORA = ("val", "val_batch01", "val_batch02", "val_kazattsd", "val_kazemo")

# Variants: distinct val-set constructions. Derived from run-name because a
# single commit (`e69a27a`) hosts both aug (exp1/2) and noaug (exp3) experiments.
VARIANT_AUG = "aug"       # batch01 aug siblings INCLUDED in val; batch02/kazattsd reals-only.
VARIANT_NOAUG = "noaug"   # all corpora reals-only.
VARIANT_LEGACY = "legacy"  # pre-exp era (04-19/04-20); val batch01 was `train_only` (reals-only).


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_git_commits() -> list[tuple[datetime, str, str]]:
    out = subprocess.check_output(
        ["git", "log", "--all", "--pretty=format:%at|%h|%s"],
        cwd=REPO_ROOT, text=True,
    )
    rows: list[tuple[datetime, str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        ts, sha, subj = line.split("|", 2)
        rows.append((datetime.fromtimestamp(int(ts)), sha, subj))
    rows.sort(key=lambda r: r[0], reverse=True)
    return rows


def commit_for_run(run_dt: datetime, commits):
    for dt, sha, subj in commits:
        if dt <= run_dt:
            return sha, subj, dt
    return None


def infer_variant(name: str, commit_sha: str | None) -> str:
    n = name or ""
    if "noaug" in n:
        return VARIANT_NOAUG
    if n.startswith("exp1-") or n.startswith("exp2-"):
        return VARIANT_AUG
    return VARIANT_LEGACY


def parse_run_name(name: str) -> dict:
    n = name or ""
    backbone = "wavlm" if "wavlm" in n else ("hubert" if "hubert" in n else "?")
    if "kazemo" in n or "-kz" in n:
        comp = "b1+b2+kz"
    elif "-b2" in n or "-01-02" in n:
        comp = "b1+b2"
    else:
        comp = "b1"
    if n.endswith("-s1") or n.endswith("-train"):
        stage = "stage1"
    elif n.endswith("-s2") or n.endswith("-ft"):
        stage = "stage2"
    elif n.endswith("-full") or "-ff" in n:
        stage = "full_ft"
    else:
        stage = "?"
    m = re.search(r"(\d{8}-\d{6})", n)
    run_dt = None
    if m:
        try:
            run_dt = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")
        except ValueError:
            pass
    return {"backbone": backbone, "composition": comp, "stage": stage, "run_dt": run_dt}


def best_over_history(history, metric, direction="max"):
    best = None
    for row in history:
        v = row.get(metric)
        if v is None:
            continue
        if best is None or (direction == "max" and v > best) or (direction == "min" and v < best):
            best = v
    return best


def extract_metrics(run) -> dict:
    history = run["history"]
    summary = run["summary"]
    out: dict = {}
    for corpus in CORPORA:
        for task in TASKS:
            acc_key = f"{corpus}/acc_{task}"
            loss_key = f"{corpus}/loss_{task}"
            out[f"best_{acc_key}"] = best_over_history(history, acc_key, "max")
            out[f"best_{loss_key}"] = best_over_history(history, loss_key, "min")
            out[f"last_{acc_key}"] = summary.get(acc_key)
            out[f"last_{loss_key}"] = summary.get(loss_key)
        total_key = f"{corpus}/loss_total"
        out[f"best_{total_key}"] = best_over_history(history, total_key, "min")
        out[f"last_{total_key}"] = summary.get(total_key)
    return out


def extract_training_time(run) -> dict:
    history = run["history"]
    summary = run["summary"]
    g_step = 0
    epoch = 0.0
    runtime = run.get("runtime_s") or summary.get("_runtime") or 0
    for row in history:
        s = row.get("_step")
        if s is not None and s > g_step:
            g_step = s
        e = row.get("train/epoch")
        if e is not None and e > epoch:
            epoch = e
        rt = row.get("_runtime")
        if rt is not None and rt > runtime:
            runtime = rt
    return {"global_step": int(g_step), "epoch": float(epoch), "runtime_s": float(runtime or 0)}


def load_runs() -> list[dict]:
    commits = load_git_commits()
    runs: list[dict] = []
    for p in sorted(EXPORT_DIR.glob("*.json")):
        if p.name == "index.json":
            continue
        r = json.loads(p.read_text())
        meta = parse_run_name(r["name"])
        commit = commit_for_run(meta["run_dt"], commits) if meta["run_dt"] else None
        commit_sha = commit[0] if commit else None
        tt = extract_training_time(r)
        runs.append({
            "id": r["id"],
            "name": r["name"],
            "state": r["state"],
            **meta,
            "commit_sha": commit_sha,
            "commit_subject": commit[1] if commit else None,
            "commit_dt": commit[2] if commit else None,
            "variant": infer_variant(r["name"], commit_sha),
            **tt,
            "metrics": extract_metrics(r),
        })
    return runs


# ── Formatting ───────────────────────────────────────────────────────────────

def fmt_num(v, pct=False):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v*100:.2f}" if pct else f"{v:.4f}"
    return str(v)


def fmt_delta(best, last):
    if best is None or last is None:
        return "—"
    return f"{last - best:+.4f}"


def fmt_runtime(s: float) -> str:
    if not s:
        return "—"
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    return f"{h}h{m:02d}m"


# ── Cohort notes ─────────────────────────────────────────────────────────────

COHORT_NOTES = {
    ("d8a2639", VARIANT_LEGACY): (
        "**04-19 hubert legacy** — first post-leakage-fix runs. Val metrics logged as "
        "concatenated `val/*` (pooled across batch01+batch02[+kazemo], no per-corpus split).\n\n"
        "Split policy: batch01 `augmented_policy: train_only` (val = reals only), batch02 "
        "`group_by: source_recording`, kazemo `valtest_all`. "
        "`class_weighting: balanced`, `num_epochs: 40` (stage2), early-stopping ON "
        "(`patience: 20`)."
    ),
    ("89a9861", VARIANT_LEGACY): (
        "**04-20 wavlm legacy** — wavlm backbone added. Val logged per-corpus "
        "(`val_batch01`, `val_batch02`, `val_kazemo`).\n\n"
        "Split policy same as 04-19. `class_weighting: balanced`, `num_epochs: 40` "
        "(stage2), early-stopping ON (`patience: 20`)."
    ),
    ("e69a27a", VARIANT_AUG): (
        "**04-21 exp1/exp2 (aug)** — hubert + wavlm × {b1, b1+b2, b1+b2+kz}, "
        "augmentation ON. Val per-corpus (`val_batch01`, `val_batch02`, `val_kazattsd`, "
        "`val_kazemo`). `val_kazattsd` is new (KazATTSD replaces KazEmo as a training "
        "corpus; KazEmo kept eval-only).\n\n"
        "Split policy: batch01 `augmented_policy: train_val_only` (aug siblings of train "
        "speakers now land in val → val_batch01 loss floor shifted ≈0.1–0.2 vs legacy), "
        "batch02 `train_only`, kazattsd `drop_all`, kazemo `valtest_all`. "
        "`class_weighting: balanced`, `num_epochs: 25` (stage2), `weight_decay: 0.05`, "
        "early-stopping OFF."
    ),
    ("e69a27a", VARIANT_NOAUG): (
        "**04-22 exp3 (noaug)** — hubert × {b1, b1+b2, b1+b2+kz}, augmentation OFF. "
        "Val per-corpus.\n\n"
        "Split policy: all training corpora `augmented_policy: drop_all` (val_batch01 back "
        "to reals-only, so NOT directly comparable to exp1/exp2 `val_batch01`); kazemo "
        "eval-only. `class_weighting: balanced`, `num_epochs: 25` (stage2), "
        "`weight_decay: 0.05`, early-stopping OFF."
    ),
}


# Cross-cohort fairness claims: which (corpus, task_or_loss) pairs share an
# identical val-data definition across which cohort keys. Used to build a
# cross-cohort leaderboard without silently comparing apples to oranges.
FAIR_ACROSS = [
    {
        "title": "`val_batch02` — reals-only across all post-leakage cohorts",
        "corpus": "val_batch02",
        "cohorts": [
            ("89a9861", VARIANT_LEGACY),
            ("e69a27a", VARIANT_AUG),
            ("e69a27a", VARIANT_NOAUG),
        ],
        "note": (
            "batch02 val never contains augmented clips: `train_only` (legacy + aug) "
            "and `drop_all` (noaug) both strip aug from val. `group_by: source_recording` "
            "is consistent across all three."
        ),
    },
    {
        "title": "`val_kazattsd` — new corpus, identical across exp1/2/3",
        "corpus": "val_kazattsd",
        "cohorts": [
            ("e69a27a", VARIANT_AUG),
            ("e69a27a", VARIANT_NOAUG),
        ],
        "note": "kazattsd is `drop_all` everywhere → val is non-aug reals only.",
    },
    {
        "title": "`val_kazemo` — eval-only, 5k balanced subsample, `valtest_all`",
        "corpus": "val_kazemo",
        "cohorts": [
            ("89a9861", VARIANT_LEGACY),
            ("e69a27a", VARIANT_AUG),
            ("e69a27a", VARIANT_NOAUG),
        ],
        "note": (
            "Same `valtest_all` speaker strategy and 5k subsample. Zero training leakage "
            "(kazemo never in train). Fairest cross-cohort generalization signal."
        ),
    },
    {
        "title": "`val_batch01` — reals-only cohorts (legacy + exp3/noaug)",
        "corpus": "val_batch01",
        "cohorts": [
            ("89a9861", VARIANT_LEGACY),
            ("e69a27a", VARIANT_NOAUG),
        ],
        "note": (
            "`train_only` (legacy) and `drop_all` (noaug) both yield reals-only val. "
            "exp1/exp2 (`train_val_only`) are NOT listed here because their val_batch01 "
            "includes aug siblings → inflated acc, deflated loss."
        ),
    },
]


# ── Report builder ───────────────────────────────────────────────────────────

def cohort_key(r) -> tuple[str, str]:
    return (r["commit_sha"] or "unknown", r["variant"])


def build_report(runs: list[dict]) -> str:
    clean_runs = [r for r in runs if r["commit_sha"] not in EXCLUDED_COMMITS]
    excluded = [r for r in runs if r["commit_sha"] in EXCLUDED_COMMITS]

    by_cohort: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in clean_runs:
        by_cohort[cohort_key(r)].append(r)

    cohort_order = sorted(
        by_cohort.keys(),
        key=lambda k: max(r["run_dt"] for r in by_cohort[k] if r["run_dt"]) if any(r["run_dt"] for r in by_cohort[k]) else datetime.min,
        reverse=True,
    )

    L: list[str] = []
    L.append("# Multi-head audio classification — results\n")
    L.append(
        f"_{len(clean_runs)} runs across {len(by_cohort)} cohorts. Cohort = "
        f"(HEAD commit at run start, split-variant). Within a cohort, val-data "
        f"definition is identical → metrics are directly comparable. "
        f"{len(excluded)} pre-leakage-fix runs excluded._\n"
    )

    # TOC
    L.append("## Contents\n")
    L.append("1. [Excluded runs](#excluded-runs)")
    L.append("2. [Cross-cohort fair leaderboards](#cross-cohort-fair-leaderboards)")
    L.append("3. [Cohort details](#cohort-details)")
    L.append("4. [Headline findings](#headline-findings)\n")

    # Excluded
    L.append("## Excluded runs\n")
    if excluded:
        L.append(
            "Runs on commit `38b23c8` (04-17). That config set "
            "`speaker_column: speaker_id` on batch02, which has no such column → "
            "splitting fell back to per-row, so augmented chunks of the same source "
            "recording could end up in train AND val simultaneously. Cross-cohort "
            "comparisons against these runs are invalid.\n"
        )
        L.append("| run | state |")
        L.append("|---|---|")
        for r in sorted(excluded, key=lambda r: r["name"]):
            L.append(f"| `{r['name']}` | {r['state']} |")
        L.append("")
    else:
        L.append("_None._\n")

    # Cross-cohort leaderboards
    L.append("## Cross-cohort fair leaderboards\n")
    L.append(
        "Each section below lists runs from cohorts where the val set for the named "
        "corpus is provably identical. Values are **best-over-history** (accuracy "
        "max, loss min). Runs are ranked by emotion accuracy within each (task, "
        "corpus) pair; the top 8 are shown. `comp` = training composition, `bb` = "
        "backbone, `var` = split variant.\n"
    )
    for spec in FAIR_ACROSS:
        pool = [r for r in clean_runs if cohort_key(r) in set(spec["cohorts"])]
        if not pool:
            continue
        L.append(f"### {spec['title']}\n")
        L.append(f"{spec['note']}\n")
        L.append(
            "| run | bb | comp | var | stage | emo acc | gen acc | age acc | loss_total |"
        )
        L.append("|---|---|---|---|---|---:|---:|---:|---:|")
        corpus = spec["corpus"]
        def sort_key(r):
            v = r["metrics"].get(f"best_{corpus}/acc_emotion")
            return -v if v is not None else 1.0
        ranked = sorted(pool, key=sort_key)
        shown = 0
        for r in ranked:
            m = r["metrics"]
            ea = m.get(f"best_{corpus}/acc_emotion")
            ga = m.get(f"best_{corpus}/acc_gender")
            aa = m.get(f"best_{corpus}/acc_age")
            lt = m.get(f"best_{corpus}/loss_total")
            if all(v is None for v in (ea, ga, aa, lt)):
                continue
            L.append(
                f"| `{r['name']}` | {r['backbone']} | {r['composition']} | "
                f"{r['variant']} | {r['stage']} | {fmt_num(ea)} | {fmt_num(ga)} | "
                f"{fmt_num(aa)} | {fmt_num(lt)} |"
            )
            shown += 1
            if shown >= 8:
                break
        L.append("")

    # Cohort details
    L.append("## Cohort details\n")
    for key in cohort_order:
        sha, variant = key
        cohort = by_cohort[key]
        if not cohort:
            continue
        subj = cohort[0]["commit_subject"] or "?"
        L.append(f"### Cohort `{sha}` · variant `{variant}` — {subj}\n")
        L.append(COHORT_NOTES.get(key, "_no notes_") + "\n")

        L.append("#### Runs in this cohort\n")
        L.append("| run | bb | comp | stage | state | global_step | epochs | runtime |")
        L.append("|---|---|---|---|---|---:|---:|---:|")
        for r in sorted(cohort, key=lambda r: (r["backbone"], r["composition"], r["stage"], r["name"])):
            L.append(
                f"| `{r['name']}` | {r['backbone']} | {r['composition']} | {r['stage']} | "
                f"{r['state']} | {r['global_step']:,} | {r['epoch']:.1f} | "
                f"{fmt_runtime(r['runtime_s'])} |"
            )
        L.append("")

        used_corpora = []
        for corpus in CORPORA:
            for r in cohort:
                m = r["metrics"]
                if any(m.get(f"best_{corpus}/acc_{t}") is not None for t in TASKS) \
                   or m.get(f"last_{corpus}/loss_total") is not None:
                    used_corpora.append(corpus)
                    break

        for corpus in used_corpora:
            L.append(f"#### `{corpus}` — best vs last\n")
            L.append(
                "| run | stage | emo best | emo last | Δ | gen best | gen last | Δ | "
                "age best | age last | Δ | loss best | loss last | Δ |"
            )
            L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for r in sorted(cohort, key=lambda r: (r["backbone"], r["composition"], r["stage"], r["name"])):
                m = r["metrics"]
                row = [f"`{r['name']}`", r["stage"]]
                has_any = False
                for task in TASKS:
                    b = m.get(f"best_{corpus}/acc_{task}")
                    l = m.get(f"last_{corpus}/acc_{task}")
                    if b is not None or l is not None:
                        has_any = True
                    row += [fmt_num(b), fmt_num(l), fmt_delta(b, l)]
                b = m.get(f"best_{corpus}/loss_total")
                l = m.get(f"last_{corpus}/loss_total")
                if b is not None or l is not None:
                    has_any = True
                row += [fmt_num(b), fmt_num(l), fmt_delta(b, l)]
                if has_any:
                    L.append("| " + " | ".join(row) + " |")
            L.append("")

    # Headline findings (data-driven, mechanical)
    L.append("## Headline findings\n")
    L.append(_headlines(clean_runs))

    return "\n".join(L)


def _headlines(clean_runs: list[dict]) -> str:
    """Auto-generated bullet list derived from `FAIR_ACROSS` leaderboards."""
    lines: list[str] = []
    for spec in FAIR_ACROSS:
        pool = [r for r in clean_runs if cohort_key(r) in set(spec["cohorts"])]
        if not pool:
            continue
        corpus = spec["corpus"]
        best: dict[str, tuple[float, str]] = {}
        for task in TASKS:
            key = f"best_{corpus}/acc_{task}"
            winner = None
            for r in pool:
                v = r["metrics"].get(key)
                if v is None:
                    continue
                if winner is None or v > winner[0]:
                    winner = (v, r["name"])
            if winner:
                best[task] = winner
        if not best:
            continue
        lines.append(f"- **{corpus}**: " + ", ".join(
            f"{t} best = {v*100:.2f}% (`{n}`)" for t, (v, n) in best.items()
        ))
    lines.append("")
    lines.append(
        "_These are the top-1 runs on each (corpus, task) among the cohort sets where "
        "val definition matches. Use the cohort-detail tables above to reason about "
        "stability (best vs last delta) and training volume._"
    )
    return "\n".join(lines)


def main():
    runs = load_runs()
    clean = [r for r in runs if r["commit_sha"] not in EXCLUDED_COMMITS]
    excluded = [r for r in runs if r["commit_sha"] in EXCLUDED_COMMITS]
    print(f"Loaded {len(runs)} runs  · clean: {len(clean)}  · excluded: {len(excluded)}")

    report = build_report(runs)
    out = OUT_DIR / "results.md"
    out.write_text(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
