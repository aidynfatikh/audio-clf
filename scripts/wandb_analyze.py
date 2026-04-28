#!/usr/bin/env python3
"""W&B analysis — post-mask-fix expN cohort.

Reads `results/wandb_export/*.json` → writes
  - `results/wandb_analysis/results.md`     (per-run + comparison tables)
  - `results/wandb_analysis/insights.md`    (paper-ready bullets)

Scope: only runs whose name starts with `exp1-` / `exp2-` AND whose
`created_at >= POST_FIX_CUTOFF`. The earlier exp1-4 runs (pre-2026-04-27) used
the broken attention-mask + unmasked mean-pool combination and are excluded.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = REPO_ROOT / "results" / "wandb_export"
OUT_DIR = REPO_ROOT / "results" / "wandb_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mask-fix landed on commit 48efd94 (2026-04-27 ~10:22 UTC). All earlier runs
# trained with broken padding handling and are not comparable to the new ones.
POST_FIX_CUTOFF = "2026-04-27T10:00:00"

TASKS = ("emotion", "gender", "age")
CORPORA = ("val_batch01", "val_batch02", "val_kazattsd", "val_kazemo")
COMPOSITIONS = ("b1", "b1+b2", "b1+b2+kz")
BACKBONES = ("hubert", "wavlm")
STAGES = ("stage1", "stage2")


def parse_name(name: str) -> dict:
    n = name or ""
    bb = "wavlm" if "wavlm" in n else ("hubert" if "hubert" in n else "?")
    if "-kz" in n:
        comp = "b1+b2+kz"
    elif "-b2" in n:
        comp = "b1+b2"
    elif "-b1" in n:
        comp = "b1"
    else:
        comp = "?"
    if n.endswith("-train"):
        stage = "stage1"
    elif n.endswith("-ft"):
        stage = "stage2"
    else:
        stage = "?"
    m = re.search(r"exp(\d+)", n)
    exp = m.group(1) if m else "?"
    return {"backbone": bb, "composition": comp, "stage": stage, "exp": exp}


def cohort_of(name: str, created_at: str) -> str:
    """post (post-mask-fix), aug (pre-fix exp1/2), noaug (pre-fix exp3), ff (pre-fix exp4)."""
    if created_at >= POST_FIX_CUTOFF:
        return "post"
    if "-noaug" in (name or ""):
        return "noaug"
    if name.startswith("exp4-") or "-ff-" in name:
        return "ff"
    return "aug"


def best_over_history(history, metric, direction="max"):
    best = None
    for row in history:
        v = row.get(metric)
        if v is None:
            continue
        if best is None or (direction == "max" and v > best) or (direction == "min" and v < best):
            best = v
    return best


def _combine(hist_v, sum_v, direction):
    if hist_v is None:
        return sum_v
    if sum_v is None:
        return hist_v
    return max(hist_v, sum_v) if direction == "max" else min(hist_v, sum_v)


def extract(run) -> dict:
    summary = run["summary"]
    history = run["history"]
    out: dict = {}
    for corpus in CORPORA:
        for task in TASKS:
            ak = f"{corpus}/acc_{task}"
            lk = f"{corpus}/loss_{task}"
            out[f"best_{ak}"] = _combine(best_over_history(history, ak, "max"), summary.get(ak), "max")
            out[f"last_{ak}"] = summary.get(ak)
            out[f"best_{lk}"] = _combine(best_over_history(history, lk, "min"), summary.get(lk), "min")
            out[f"last_{lk}"] = summary.get(lk)
        out[f"last_{corpus}/loss_total"] = summary.get(f"{corpus}/loss_total")
    out["best_val/score"] = _combine(
        best_over_history(history, "val/score", "max"), summary.get("val/score"), "max"
    )
    out["last_val/score"] = summary.get("val/score")
    out["last_val/loss_total"] = summary.get("val/loss_total")
    return out


def load_runs(*, post_only: bool = True) -> list[dict]:
    runs: list[dict] = []
    for p in sorted(EXPORT_DIR.glob("*.json")):
        if p.name == "index.json":
            continue
        r = json.loads(p.read_text())
        name = r.get("name") or ""
        if not name.startswith(("exp1-", "exp2-", "exp3-", "exp4-")):
            continue
        created = r.get("created_at") or ""
        coh = cohort_of(name, created)
        if post_only and coh != "post":
            continue
        meta = parse_name(name)
        runs.append({
            "id": r["id"],
            "name": name,
            "state": r["state"],
            "created_at": created,
            "runtime_s": float(r.get("runtime_s") or 0),
            "cohort": coh,
            **meta,
            "metrics": extract(r),
        })
    def sort_key(r):
        comp_idx = COMPOSITIONS.index(r["composition"]) if r["composition"] in COMPOSITIONS else 99
        return (r["cohort"], r["backbone"], comp_idx, r["stage"])
    runs.sort(key=sort_key)
    return runs


# ── Formatting ──────────────────────────────────────────────────────────────

def pct(v):
    return "—" if v is None else f"{v*100:.2f}"


def num(v):
    return "—" if v is None else f"{v:.4f}"


def runtime(s: float) -> str:
    if not s:
        return "—"
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    return f"{h}h{m:02d}m"


def find(runs, *, backbone, composition, stage):
    for r in runs:
        if r["backbone"] == backbone and r["composition"] == composition and r["stage"] == stage:
            return r
    return None


# ── Report ──────────────────────────────────────────────────────────────────

def section_runs_table(runs):
    L = ["### Runs in this cohort\n",
         "| run | bb | composition | stage | state | runtime | val/score (best) |",
         "|---|---|---|---|---|---:|---:|"]
    for r in runs:
        L.append(
            f"| `{r['name']}` | {r['backbone']} | {r['composition']} | {r['stage']} | "
            f"{r['state']} | {runtime(r['runtime_s'])} | {pct(r['metrics']['best_val/score'])} |"
        )
    L.append("")
    return "\n".join(L)


def section_per_corpus_leaderboard(runs):
    """For each corpus, rank stage2 (ft) runs by per-task best accuracy."""
    L = ["## Per-corpus stage-2 leaderboard\n",
         "Best-over-history validation accuracy per task. Stage-2 (fine-tuned) runs only. "
         "`val_kazemo` is held-out (never seen at training time) and is the cleanest "
         "generalisation signal.\n"]
    ft = [r for r in runs if r["stage"] == "stage2"]
    for corpus in CORPORA:
        L.append(f"### `{corpus}`\n")
        L.append("| bb | composition | emo | gen | age |")
        L.append("|---|---|---:|---:|---:|")
        rows = []
        for r in ft:
            m = r["metrics"]
            ea = m.get(f"best_{corpus}/acc_emotion")
            ga = m.get(f"best_{corpus}/acc_gender")
            aa = m.get(f"best_{corpus}/acc_age")
            if ea is None and ga is None and aa is None:
                continue
            rows.append((r, ea, ga, aa))
        rows.sort(key=lambda x: -(x[1] or 0))
        for r, ea, ga, aa in rows:
            L.append(f"| {r['backbone']} | {r['composition']} | {pct(ea)} | {pct(ga)} | {pct(aa)} |")
        L.append("")
    return "\n".join(L)


def section_stage1_vs_stage2(runs):
    """Within each (bb, comp), how much does fine-tuning add over a frozen backbone?"""
    L = ["## Effect of fine-tuning (stage 1 → stage 2)\n",
         "Δ accuracy from unfreezing the backbone. Reported per (corpus, task) on the "
         "best-over-history value. Positive Δ ⇒ fine-tuning helped.\n"]
    for bb in BACKBONES:
        for comp in COMPOSITIONS:
            s1 = find(runs, backbone=bb, composition=comp, stage="stage1")
            s2 = find(runs, backbone=bb, composition=comp, stage="stage2")
            if not s1 or not s2:
                continue
            L.append(f"### `{bb}` · `{comp}`\n")
            L.append("| corpus | task | s1 | s2 | Δ (pp) |")
            L.append("|---|---|---:|---:|---:|")
            for corpus in CORPORA:
                for task in TASKS:
                    k = f"best_{corpus}/acc_{task}"
                    a, b = s1["metrics"].get(k), s2["metrics"].get(k)
                    if a is None and b is None:
                        continue
                    if a is None or b is None:
                        L.append(f"| {corpus} | {task} | {pct(a)} | {pct(b)} | — |")
                    else:
                        L.append(f"| {corpus} | {task} | {pct(a)} | {pct(b)} | {(b-a)*100:+.2f} |")
            L.append("")
    return "\n".join(L)


def section_composition_effect(runs):
    """For each (bb, stage), how does adding b2 / kazattsd shift each corpus?"""
    L = ["## Effect of training composition\n",
         "How adding more training corpora moves stage-2 validation accuracy on each "
         "evaluation corpus. `b1` = batch01 only; `b1+b2` adds batch02; `b1+b2+kz` "
         "adds KazATTSD. KazEmo is never in train.\n"]
    for bb in BACKBONES:
        L.append(f"### `{bb}` (stage 2)\n")
        L.append("| corpus | task | b1 | b1+b2 | b1+b2+kz |")
        L.append("|---|---|---:|---:|---:|")
        for corpus in CORPORA:
            for task in TASKS:
                k = f"best_{corpus}/acc_{task}"
                vals = []
                for comp in COMPOSITIONS:
                    r = find(runs, backbone=bb, composition=comp, stage="stage2")
                    vals.append(r["metrics"].get(k) if r else None)
                if all(v is None for v in vals):
                    continue
                L.append(f"| {corpus} | {task} | " + " | ".join(pct(v) for v in vals) + " |")
        L.append("")
    return "\n".join(L)


def section_backbone_compare(runs):
    L = ["## Backbone comparison (HuBERT vs WavLM)\n",
         "Stage-2 best validation accuracy at matched composition.\n"]
    for comp in COMPOSITIONS:
        L.append(f"### Composition `{comp}` (stage 2)\n")
        L.append("| corpus | task | hubert | wavlm | Δ (pp) |")
        L.append("|---|---|---:|---:|---:|")
        for corpus in CORPORA:
            for task in TASKS:
                k = f"best_{corpus}/acc_{task}"
                rh = find(runs, backbone="hubert", composition=comp, stage="stage2")
                rw = find(runs, backbone="wavlm", composition=comp, stage="stage2")
                vh = rh["metrics"].get(k) if rh else None
                vw = rw["metrics"].get(k) if rw else None
                if vh is None and vw is None:
                    continue
                if vh is None or vw is None:
                    L.append(f"| {corpus} | {task} | {pct(vh)} | {pct(vw)} | — |")
                else:
                    L.append(f"| {corpus} | {task} | {pct(vh)} | {pct(vw)} | {(vw-vh)*100:+.2f} |")
        L.append("")
    return "\n".join(L)


METHODOLOGY = """## Methodology

### Task

Single shared encoder, three classification heads trained jointly:

- **Emotion** — 7 classes (`angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, `surprised`).
- **Gender** — 2 classes (`F`, `M`).
- **Age category** — 4 classes (`child`, `young`, `adult`, `senior`).

Total loss is the unweighted sum of per-task cross-entropies; rows missing a
label for a task contribute zero to that task's loss (label sentinel `-100`).
Cross-entropy uses sklearn-style `balanced` class weights computed from the
training split, so a rare class's total gradient mass equals a common one's.
Label smoothing 0.1.

### Corpora

| corpus | role | source | notes |
|---|---|---|---|
| `batch01` | train + eval | `01gumano1d/batch01-validation-test` | small validation/test corpus, full label coverage |
| `batch02` | train + eval | `01gumano1d/batch2-aug-clean` | largest training corpus; rows grouped by `source_recording` so augmented siblings cannot cross splits |
| `kazattsd` | train + eval | `01gumano1d/KazATTSD-batch0` | scripted/TTS-style corpus |
| `kazemo` | **eval only** | local cache | held out completely from training; 5 k balanced subsample, `valtest_all` strategy |

### Splits

Splits are produced by `splits/builder.py` from a YAML config and frozen as
parquet manifests. Each split is **speaker-disjoint** and stratified by
emotion. For batch02 the disjointness key is `source_recording` rather than
speaker_id.

Three training compositions are evaluated:

- `b1` — train on batch01 only.
- `b1+b2` — train on batch01 + batch02.
- `b1+b2+kz` — train on batch01 + batch02 + KazATTSD.

A **single shared evaluation manifest** (`combined_eval_v1`) provides the
val/test rows for every run. Every corpus contributes to it with
`augmented_policy: drop_all`, so the eval pool is reals-only across the board
and the same rows are scored across compositions and backbones. KazEmo is
routed `valtest_all` → 5 k balanced subsample, never in train.

A speaker-leakage guard at training time (`utils/data_loading._check_speaker_leakage`)
fails fast if any (dataset, speaker_id) pair appears in both the per-experiment
train manifest and the shared eval manifest.

### Audio preprocessing

- Decode → mono → resample to **16 kHz** with polyphase resampling.
- Truncate or zero-pad each clip to a fixed **10 s** window (160 000 samples).
- Pre-pad raw length is recorded and propagated to the model as `input_length`
  so the pooling stage can ignore the padded tail (see Architecture).
- Train-only augmentation (when enabled by the split config): SoX-style speed
  perturbation `{0.9, 1.0, 1.1}` with `p=0.8`, and additive background noise
  mixed at SNR `U(5, 20) dB` with `p=0.5`. Augmentation is applied online —
  no augmented copies are stored in manifests.

### Architecture

`MultiTaskBackbone` (`multihead/model.py`):

- Backbone: `facebook/hubert-base-ls960` or `microsoft/wavlm-base-plus`,
  HF transformers, `output_hidden_states=True` (13 layers including the
  pre-transformer projection).
- **No `attention_mask` is passed to the backbone.** Both base variants use
  `feat_extract_norm="group"` and were pretrained without a mask; passing one
  shifts the pretrained activation distribution. (For `large` variants this
  is the opposite — they would need the mask.)
- Per-task **softmax-weighted layer sum**: each head holds a learnable vector
  over the 13 layers, softmaxed and applied as a weighted sum to produce a
  task-specific `[B, T, H]` representation.
- **Masked mean-pool** over the time axis. The pad mask is built from the
  per-row pre-pad sample length via `backbone._get_feat_extract_output_lengths`,
  which applies the conv stem's stride formula to map sample-counts → frame-counts.
  Padded frames contribute zero to numerator and denominator, so silence does
  not dilute the embedding.
- Three classification heads: `Linear(H, 256) → ReLU → Dropout → Linear(256, K)`,
  with H = backbone hidden size (768 for base) and K = task class count.

### Two-stage training

Every (backbone, composition) pair runs `stage1 → stage2` sequentially.

**Stage 1 — head-only.** Backbone frozen (`freeze_backbone: true`), only the
layer weights and three heads receive gradients. AdamW, head LR `1e-3`,
weight decay `0.01`, batch size 4, **15 epochs**, cosine schedule (3-epoch
hold + 12-epoch decay to `1e-2 · base`), grad clip 1.0. Best stage-1
checkpoint by `val/score` is fed into stage 2.

**Stage 2 — partial unfreeze + LLRD.** Heads keep training; the **top 3
contiguous transformer layers** of the backbone are unfrozen (no gaps in the
gradient flow). Backbone LR `5e-5` at the top layer with **layer-wise LR
decay 0.85**, head LR `2e-4`, weight decay `0.05`, batch size 8,
**25 epochs**, cosine schedule (3-hold + 22-decay) with per-group power
scaling so the top (highest-LR) groups decay slightly faster than the
bottom. SpecAugment is enabled at this stage (HF built-in,
`mask_time_prob=0.05`, `mask_time_length=10`). Drop-path 0.15 inside the
backbone. Grad clip 1.0. Early stopping disabled — the cosine schedule
defines the run length.

### Validation & best-model selection

Validation runs **5 times per epoch** on the shared eval manifest, with
metrics logged per corpus (`val_batch01`, `val_batch02`, `val_kazattsd`,
`val_kazemo`) plus a pooled `val/*`.

Best-model criterion is `val/score` — **the macro-mean of per-(corpus, task)
accuracies** on the shared val set: per corpus, average the available task
accuracies; then average across corpora. This weights every (corpus, task)
cell equally and resists imbalanced corpora dominating the model selection.

### What changed for this cohort

All runs reported here are post-commit `48efd94` (2026-04-27). Compared to
earlier exp1–exp4 runs, three things changed:

1. **No `attention_mask` to the backbone** (was being supplied incorrectly).
2. **Masked mean-pool over time** using `_get_feat_extract_output_lengths`
   (previously an unmasked mean → silence padding diluted every embedding).
3. **Shared eval manifest + macro-mean accuracy** for both validation and
   best-model selection (previously per-experiment val pools and val-loss).

Older runs are NOT included in this report because they are not directly
comparable — eval pools and pooling correctness both differ.

"""


def build_report(runs):
    L = ["# Post-mask-fix experiments — results\n"]
    L.append(
        f"_{len(runs)} runs from the post-mask-fix cohort (commit 48efd94, 2026-04-27)._\n\n"
        "All runs use the same shared eval manifest (`splits/combined_eval_v1`) so val "
        "and test sets are identical across compositions and backbones. Best-model "
        "selection tracks macro-mean accuracy (`val/score`).\n"
    )
    L.append("## Contents\n")
    L.append("1. [Methodology](#methodology)")
    L.append("2. [Runs in this cohort](#runs-in-this-cohort)")
    L.append("3. [Per-corpus stage-2 leaderboard](#per-corpus-stage-2-leaderboard)")
    L.append("4. [Effect of fine-tuning (stage 1 → stage 2)](#effect-of-fine-tuning-stage-1--stage-2)")
    L.append("5. [Effect of training composition](#effect-of-training-composition)")
    L.append("6. [Backbone comparison (HuBERT vs WavLM)](#backbone-comparison-hubert-vs-wavlm)\n")

    L.append(METHODOLOGY)
    L.append(section_runs_table(runs))
    L.append(section_per_corpus_leaderboard(runs))
    L.append(section_stage1_vs_stage2(runs))
    L.append(section_composition_effect(runs))
    L.append(section_backbone_compare(runs))
    return "\n".join(L)


# ── Insights ────────────────────────────────────────────────────────────────

def build_insights(runs):
    """Quotable, paper-ready bullets derived directly from the leaderboard."""
    ft = [r for r in runs if r["stage"] == "stage2"]

    def best_for(corpus, task):
        winner = None
        for r in ft:
            v = r["metrics"].get(f"best_{corpus}/acc_{task}")
            if v is None:
                continue
            if winner is None or v > winner[0]:
                winner = (v, r)
        return winner

    L = ["# Paper-ready insights\n"]
    L.append("_Derived from the post-mask-fix cohort. All numbers are best-over-history "
             "validation accuracy on the shared eval manifest. Pair this file with "
             "`results.md` for the full methodology and tables._\n")

    L.append("## Setup at a glance\n")
    L.append(
        "- **Task:** multi-head classification of emotion (7), gender (2), and age "
        "category (4) on Kazakh speech, joint loss over the three heads.\n"
        "- **Training corpora:** batch01, batch02, KazATTSD. Three compositions — "
        "`b1`, `b1+b2`, `b1+b2+kz` — × two backbones (HuBERT-base, WavLM-base+).\n"
        "- **Held-out:** KazEmo, never seen at training time. The same shared eval "
        "manifest (`combined_eval_v1`) is used by every run, so val and test sets "
        "are identical across compositions and backbones.\n"
        "- **Model:** SSL backbone with three task-specific weighted layer sums and "
        "a masked mean-pool over the time axis. The pad mask is computed from the "
        "raw sample length via `backbone._get_feat_extract_output_lengths`; the base "
        "backbone is run without an `attention_mask` to match its pretraining regime.\n"
        "- **Two-stage training:** stage 1 = frozen backbone + heads (15 epochs, "
        "AdamW 1e-3, batch 4, cosine 3+12); stage 2 = partial unfreeze of the top 3 "
        "transformer layers with layer-wise LR decay 0.85, backbone LR 5e-5 / head "
        "LR 2e-4, weight decay 0.05, drop-path 0.15, SpecAugment, batch 8, 25 "
        "epochs, cosine 3+22, no early stopping.\n"
        "- **Best-model criterion:** macro-mean of per-(corpus, task) validation "
        "accuracies (`val/score`).\n"
    )
    L.append("")

    L.append("## Headline numbers\n")
    for corpus in CORPORA:
        bullets = []
        for task in TASKS:
            w = best_for(corpus, task)
            if w is None:
                continue
            v, r = w
            bullets.append(f"{task} **{v*100:.2f}%** ({r['backbone']} · {r['composition']})")
        if bullets:
            L.append(f"- **{corpus}** — " + "; ".join(bullets))
    L.append("")

    # KazEmo zero-shot generalisation
    L.append("## Generalisation to held-out KazEmo (never in train)\n")
    L.append("| bb | composition | KazEmo emo acc |")
    L.append("|---|---|---:|")
    for bb in BACKBONES:
        for comp in COMPOSITIONS:
            r = find(runs, backbone=bb, composition=comp, stage="stage2")
            if not r:
                continue
            v = r["metrics"].get("best_val_kazemo/acc_emotion")
            L.append(f"| {bb} | {comp} | {pct(v)} |")
    L.append("")
    L.append(
        "_KazEmo is a separate corpus held out completely from training. Its emotion "
        "accuracy is the closest thing in this cohort to a true cross-corpus generalisation "
        "score for Kazakh speech-emotion recognition._\n"
    )

    # Stage1 vs Stage2 magnitude
    L.append("## Fine-tuning lift (stage 2 over stage 1)\n")
    deltas = []
    for bb in BACKBONES:
        for comp in COMPOSITIONS:
            s1 = find(runs, backbone=bb, composition=comp, stage="stage1")
            s2 = find(runs, backbone=bb, composition=comp, stage="stage2")
            if not s1 or not s2:
                continue
            for corpus in CORPORA:
                for task in TASKS:
                    k = f"best_{corpus}/acc_{task}"
                    a = s1["metrics"].get(k)
                    b = s2["metrics"].get(k)
                    if a is None or b is None:
                        continue
                    deltas.append((bb, comp, corpus, task, b - a))
    if deltas:
        emo = [d for d in deltas if d[3] == "emotion"]
        gen = [d for d in deltas if d[3] == "gender"]
        age = [d for d in deltas if d[3] == "age"]
        L.append(
            f"- Mean Δ on **emotion**: {sum(d[4] for d in emo)/len(emo)*100:+.2f} pp "
            f"(N={len(emo)})"
        )
        L.append(
            f"- Mean Δ on **gender**: {sum(d[4] for d in gen)/len(gen)*100:+.2f} pp "
            f"(N={len(gen)})"
        )
        L.append(
            f"- Mean Δ on **age**: {sum(d[4] for d in age)/len(age)*100:+.2f} pp "
            f"(N={len(age)})"
        )
    L.append("")

    # Composition effect summary
    L.append("## Composition effect (stage 2)\n")
    for bb in BACKBONES:
        for corpus in CORPORA:
            row = []
            for comp in COMPOSITIONS:
                r = find(runs, backbone=bb, composition=comp, stage="stage2")
                v = r["metrics"].get(f"best_{corpus}/acc_emotion") if r else None
                row.append((comp, v))
            if all(v is None for _, v in row):
                continue
            seq = " → ".join(f"{c} {pct(v)}" for c, v in row)
            L.append(f"- `{bb}` · {corpus} emotion: {seq}")
    L.append("")
    L.append(
        "_Reading the rows: a monotone increase shows that wider training composition "
        "helps the named eval corpus; a peak at `b1+b2` followed by a drop at `b1+b2+kz` "
        "shows that adding KazATTSD hurts that particular evaluation slice._\n"
    )

    # Backbone summary
    L.append("## HuBERT vs WavLM (stage 2)\n")
    wins = {"hubert": 0, "wavlm": 0, "tie": 0}
    for comp in COMPOSITIONS:
        rh = find(runs, backbone="hubert", composition=comp, stage="stage2")
        rw = find(runs, backbone="wavlm", composition=comp, stage="stage2")
        if not rh or not rw:
            continue
        for corpus in CORPORA:
            for task in TASKS:
                k = f"best_{corpus}/acc_{task}"
                vh = rh["metrics"].get(k)
                vw = rw["metrics"].get(k)
                if vh is None or vw is None:
                    continue
                if abs(vh - vw) < 1e-4:
                    wins["tie"] += 1
                elif vw > vh:
                    wins["wavlm"] += 1
                else:
                    wins["hubert"] += 1
    total = sum(wins.values())
    if total:
        L.append(
            f"- Across {total} (composition, corpus, task) head-to-heads: "
            f"WavLM wins **{wins['wavlm']}**, HuBERT wins **{wins['hubert']}**, "
            f"ties {wins['tie']}."
        )
    L.append("")
    return "\n".join(L)


def find_in(pool, *, backbone, composition, stage):
    for r in pool:
        if r["backbone"] == backbone and r["composition"] == composition and r["stage"] == stage:
            return r
    return None


COHORT_LABEL = {
    "post":  "post-fix (exp1/2 · 2026-04-27, aug splits, masked pool)",
    "aug":   "pre-fix exp1/2 (2026-04-21, aug splits, broken mask + unmasked pool)",
    "noaug": "pre-fix exp3 (2026-04-22, noaug splits, broken mask + unmasked pool)",
    "ff":    "pre-fix exp4 (2026-04-22, full-finetune, broken mask + unmasked pool)",
}


def build_comparison(all_runs):
    """Side-by-side per-(corpus, task) stage-2 table across cohorts + consistency notes.

    Pre-fix and post-fix runs use *different* val pools: pre-fix used per-experiment
    val sets while post-fix uses the shared `combined_eval_v1` manifest. Absolute
    accuracy values are therefore NOT directly comparable across cohorts. The
    interesting question is whether *qualitative trends* (composition effect,
    backbone ranking, KazEmo zero-shot direction) are stable.
    """
    cohorts = ("post", "aug", "noaug")
    by_cohort = {c: [r for r in all_runs if r["cohort"] == c] for c in cohorts}

    L = ["# Pre-fix vs post-fix comparison\n"]
    L.append(
        "_All expN runs (pre-mask-fix and post-mask-fix). Pre-fix runs trained with "
        "the broken `attention_mask` + unmasked-mean-pool combination and used "
        "per-experiment val pools, so **absolute numbers are not comparable across "
        "cohorts**. The point of this file is to check whether the **qualitative "
        "conclusions** survive the bug-fix._\n"
    )

    # 1. Cohort summary
    L.append("## Cohorts in this analysis\n")
    L.append("| cohort | description | runs |")
    L.append("|---|---|---:|")
    for c in cohorts:
        L.append(f"| `{c}` | {COHORT_LABEL[c]} | {len(by_cohort[c])} |")
    L.append("")

    # 2. Side-by-side stage-2 emotion table per (bb, composition, corpus)
    L.append("## Stage-2 emotion accuracy per cohort\n")
    L.append(
        "Best-over-history validation emotion accuracy on each evaluation corpus, "
        "for each (backbone, composition). Empty cell ⇒ that cohort did not run "
        "this configuration.\n"
    )
    for bb in BACKBONES:
        L.append(f"### `{bb}` — emotion (stage 2)\n")
        L.append("| corpus | composition | post | aug (pre) | noaug (pre) |")
        L.append("|---|---|---:|---:|---:|")
        for corpus in CORPORA:
            for comp in COMPOSITIONS:
                vals = []
                for c in cohorts:
                    r = find_in(by_cohort[c], backbone=bb, composition=comp, stage="stage2")
                    v = r["metrics"].get(f"best_{corpus}/acc_emotion") if r else None
                    vals.append(v)
                if all(v is None for v in vals):
                    continue
                L.append(
                    f"| {corpus} | {comp} | " + " | ".join(pct(v) for v in vals) + " |"
                )
        L.append("")

    # 3. Composition effect direction — sign-only check
    L.append("## Composition-effect direction (consistency check)\n")
    L.append(
        "Sign of `Δ_emotion = acc(b1+b2+kz) − acc(b1)` on each (backbone, corpus) "
        "in each cohort. **+** = adding more training data helped, **−** = hurt, "
        "**·** = unavailable. The story to look for is whether the same sign "
        "appears across cohorts on the same row — that's the bug-independent "
        "signal.\n"
    )
    L.append("| corpus | bb | post | aug | noaug |")
    L.append("|---|---|:-:|:-:|:-:|")
    for bb in BACKBONES:
        for corpus in CORPORA:
            cells = []
            for c in cohorts:
                pool = by_cohort[c]
                r1 = find_in(pool, backbone=bb, composition="b1", stage="stage2")
                r3 = find_in(pool, backbone=bb, composition="b1+b2+kz", stage="stage2")
                v1 = r1["metrics"].get(f"best_{corpus}/acc_emotion") if r1 else None
                v3 = r3["metrics"].get(f"best_{corpus}/acc_emotion") if r3 else None
                if v1 is None or v3 is None:
                    cells.append("·")
                else:
                    cells.append(f"+{(v3-v1)*100:.1f}" if v3 >= v1 else f"−{(v1-v3)*100:.1f}")
            L.append(f"| {corpus} | {bb} | " + " | ".join(cells) + " |")
    L.append("")

    # 4. Backbone ranking direction
    L.append("## Backbone direction (HuBERT vs WavLM)\n")
    L.append(
        "Sign of `Δ_emotion = wavlm − hubert` on stage-2 emotion accuracy per "
        "(composition, corpus). **+** = WavLM higher, **−** = HuBERT higher.\n"
    )
    L.append("| corpus | comp | post | aug | noaug |")
    L.append("|---|---|:-:|:-:|:-:|")
    for corpus in CORPORA:
        for comp in COMPOSITIONS:
            cells = []
            for c in cohorts:
                pool = by_cohort[c]
                rh = find_in(pool, backbone="hubert", composition=comp, stage="stage2")
                rw = find_in(pool, backbone="wavlm",  composition=comp, stage="stage2")
                vh = rh["metrics"].get(f"best_{corpus}/acc_emotion") if rh else None
                vw = rw["metrics"].get(f"best_{corpus}/acc_emotion") if rw else None
                if vh is None or vw is None:
                    cells.append("·")
                else:
                    cells.append(f"+{(vw-vh)*100:.1f}" if vw >= vh else f"−{(vh-vw)*100:.1f}")
            L.append(f"| {corpus} | {comp} | " + " | ".join(cells) + " |")
    L.append("")

    # 5. KazEmo zero-shot trend
    L.append("## KazEmo (held-out) — direction of composition effect\n")
    L.append(
        "Stage-2 KazEmo emotion accuracy across compositions, per cohort. KazEmo is "
        "never in train under any cohort. The post-fix observation is that adding "
        "more training data (b2 / KazATTSD) **hurts** KazEmo — does the pre-fix "
        "data show the same?\n"
    )
    for bb in BACKBONES:
        L.append(f"### `{bb}`\n")
        L.append("| cohort | b1 | b1+b2 | b1+b2+kz | direction |")
        L.append("|---|---:|---:|---:|:-:|")
        for c in cohorts:
            pool = by_cohort[c]
            row_vals = []
            for comp in COMPOSITIONS:
                r = find_in(pool, backbone=bb, composition=comp, stage="stage2")
                v = r["metrics"].get("best_val_kazemo/acc_emotion") if r else None
                row_vals.append(v)
            if all(v is None for v in row_vals):
                continue
            v1, v2, v3 = row_vals
            if v1 is not None and v3 is not None:
                if v3 > v1:
                    direction = f"↑ +{(v3-v1)*100:.1f} pp"
                elif v3 < v1:
                    direction = f"↓ −{(v1-v3)*100:.1f} pp"
                else:
                    direction = "—"
            else:
                direction = "·"
            L.append(
                f"| {c} | " + " | ".join(pct(v) for v in row_vals) + f" | {direction} |"
            )
        L.append("")

    # 6. Consistency summary
    L.append("## What survives the bug-fix\n")
    L.append(_consistency_summary(by_cohort))
    return "\n".join(L)


def _consistency_summary(by_cohort) -> str:
    """Build a bullet list of what's consistent and what isn't."""
    cohorts = ("post", "aug", "noaug")

    def comp_sign(c, bb, corpus):
        pool = by_cohort[c]
        r1 = find_in(pool, backbone=bb, composition="b1", stage="stage2")
        r3 = find_in(pool, backbone=bb, composition="b1+b2+kz", stage="stage2")
        v1 = r1["metrics"].get(f"best_{corpus}/acc_emotion") if r1 else None
        v3 = r3["metrics"].get(f"best_{corpus}/acc_emotion") if r3 else None
        if v1 is None or v3 is None:
            return None
        return "+" if v3 >= v1 else "-"

    consistent = []
    inconsistent = []
    for bb in BACKBONES:
        for corpus in CORPORA:
            signs = [comp_sign(c, bb, corpus) for c in cohorts]
            present = [s for s in signs if s is not None]
            if len(present) < 2:
                continue
            label = f"`{bb}` · {corpus}: " + " / ".join(
                f"{c}={s if s else '·'}" for c, s in zip(cohorts, signs)
            )
            if len(set(present)) == 1:
                consistent.append(label)
            else:
                inconsistent.append(label)

    L = []
    L.append("**Composition direction `b1 → b1+b2+kz` (emotion, stage 2).** "
             "Pre-fix b1-only runs only logged `val_batch01` (the per-experiment val "
             "pool was restricted to training corpora) so this endpoint check is "
             "limited to that corpus.\n")
    if consistent:
        L.append("- _Same sign across every cohort with data:_")
        for x in consistent:
            L.append(f"  - {x}")
    if inconsistent:
        L.append("- _Direction **flips** between cohorts (treat with care):_")
        for x in inconsistent:
            L.append(f"  - {x}")
    if not consistent and not inconsistent:
        L.append("- _No (bb, corpus) row has both endpoints in any pre-fix cohort._")
    L.append("")

    # Composition direction at the b1+b2 → b1+b2+kz endpoint, which IS available in
    # all three cohorts. Adding KazATTSD on top of b1+b2 — does it help or hurt?
    L.append("**Composition direction `b1+b2 → b1+b2+kz` (emotion, stage 2).** "
             "Both endpoints are available in every cohort, so this is the cleanest "
             "consistency check. **+** = KazATTSD helped, **−** = hurt.\n")
    L.append("| corpus | bb | post | aug | noaug |")
    L.append("|---|---|:-:|:-:|:-:|")
    for bb in BACKBONES:
        for corpus in CORPORA:
            cells = []
            for c in cohorts:
                pool = by_cohort[c]
                r2 = find_in(pool, backbone=bb, composition="b1+b2", stage="stage2")
                r3 = find_in(pool, backbone=bb, composition="b1+b2+kz", stage="stage2")
                v2 = r2["metrics"].get(f"best_{corpus}/acc_emotion") if r2 else None
                v3 = r3["metrics"].get(f"best_{corpus}/acc_emotion") if r3 else None
                if v2 is None or v3 is None:
                    cells.append("·")
                else:
                    cells.append(f"+{(v3-v2)*100:.1f}" if v3 >= v2 else f"−{(v2-v3)*100:.1f}")
            if all(c == "·" for c in cells):
                continue
            L.append(f"| {corpus} | {bb} | " + " | ".join(cells) + " |")
    L.append("")

    # KazEmo zero-shot direction at b1+b2 → b1+b2+kz
    L.append("**KazEmo zero-shot (held out).** Direction of "
             "`b1+b2+kz − b1+b2` on stage-2 KazEmo emotion accuracy:\n")
    for bb in BACKBONES:
        parts = []
        for c in cohorts:
            pool = by_cohort[c]
            r2 = find_in(pool, backbone=bb, composition="b1+b2", stage="stage2")
            r3 = find_in(pool, backbone=bb, composition="b1+b2+kz", stage="stage2")
            v2 = r2["metrics"].get("best_val_kazemo/acc_emotion") if r2 else None
            v3 = r3["metrics"].get("best_val_kazemo/acc_emotion") if r3 else None
            if v2 is None or v3 is None:
                parts.append(f"{c}=·")
            else:
                parts.append(f"{c}={(v3-v2)*100:+.1f} pp")
        L.append(f"- `{bb}`: " + ", ".join(parts))
    L.append("")
    L.append(
        "_Interpretation: if all three cohorts show the same sign on KazEmo, the "
        "'KazATTSD doesn't help (and often hurts) held-out KazEmo' finding is a "
        "real data-distribution effect, not an artefact of the masking bug._"
    )
    return "\n".join(L)


def main():
    runs = load_runs(post_only=True)
    print(f"Loaded {len(runs)} post-fix expN runs")
    for r in runs:
        print(f"  · {r['name']}  [{r['state']}]")
    (OUT_DIR / "results.md").write_text(build_report(runs))
    (OUT_DIR / "insights.md").write_text(build_insights(runs))

    all_runs = load_runs(post_only=False)
    print(f"Loaded {len(all_runs)} expN runs total (post + pre-fix) for comparison")
    (OUT_DIR / "pre_fix_comparison.md").write_text(build_comparison(all_runs))
    print(f"Wrote {OUT_DIR/'results.md'}, {OUT_DIR/'insights.md'}, {OUT_DIR/'pre_fix_comparison.md'}")


if __name__ == "__main__":
    main()
