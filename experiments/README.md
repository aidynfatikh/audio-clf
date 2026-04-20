# Experiments — splits and runs

This doc covers two things: (1) how a YAML split config becomes train/val/test
manifests with no leakage, and (2) what each of the four experiments does and
why.

## 1. Split building

### 1.1 Entry point

```
python scripts/build_splits.py --config configs/splits/<name>.yaml
```

Produces under `splits/<name>/`:

- `train.parquet`, `val.parquet`, `test.parquet` — per-split row manifests
- `summary.json` — totals, per-corpus/per-emotion counts, leak-check report,
  config hash, git sha

Each experiment shell script calls this builder once per cell before
training, and re-uses the output directory on subsequent runs.

### 1.2 Corpora

Three sources, each toggleable per config:

| corpus  | HF id                                   | group key                     | notes                                 |
|---------|-----------------------------------------|-------------------------------|---------------------------------------|
| batch01 | `01gumano1d/batch01-validation-test`    | `speaker_id`                  | has augmented siblings                |
| batch02 | `01gumano1d/batch2-aug-clean`           | `original_name`               | chunked; aug siblings share recording |
| kazemo  | local (`data/kazemo`)                   | `speaker_id` (3 speakers)     | emotion-only                          |

Each row is normalized into a `NormalizedRow` carrying `dataset`, `row_id`,
`speaker_id` (doubles as the group key), `emotion`/`gender`/`age` labels, and
an `augmented: bool` flag.

### 1.3 Pipeline

```
YAML  →  sources.load_corpus  →  builder._split_corpus  →  leak_check  →  write manifests
```

1. **`load_corpus` (splits/sources.py)** — downloads/caches the HF dataset
   metadata (no audio bytes), normalizes each row, and attaches the
   `augmented` flag (from `augmented` / `is_augmented` / `metadata` column,
   whichever exists). If `augmented_policy: drop_all`, aug rows are dropped
   here.

2. **`_split_corpus` (splits/builder.py)** — splits the *non-augmented* pool
   with a stratified group 3-way split (stratify by `emotion`, group by
   `speaker_id` / `original_name`), then routes aug rows per policy.

3. **`_compute_leak_check` (splits/builder.py)** — verifies every aug row sits
   only in a split permitted by its per-corpus policy. Writes the result to
   `summary.json.leak_check`. Aborts on any violation.

4. **Manifests written** (parquet per split) plus `summary.json`.

### 1.4 Augmentation policies

Configured per-corpus as `augmented_policy: <value>`:

| policy             | aug rows end up in           | typical use                                     |
|--------------------|------------------------------|-------------------------------------------------|
| `drop_all`         | nowhere (filtered at load)   | Exp 3 (clean-only runs)                         |
| `train_only`       | train (if parent in train)   | batch02 in Exp 1/2 — aug copies stay with train |
| `train_val_only`   | train or val (not test)      | batch01 in Exp 1/2 — val keeps aug+clean mix    |
| `keep_all`         | wherever the split ends up   | default — no leakage guarantees                 |

**How routing works.** A group key (speaker_id for batch01, `original_name`
for batch02) determines which split its non-aug rows landed in. Each aug row
is attached to its parent's group key and routed to that same split — but
only if the split is in `allowed_splits`. If the parent ended up in a
disallowed split (e.g. test under `train_val_only`), the aug row is dropped
and accounted for in `summary.json`.

Result, guaranteed:

- **No aug row can appear in a split whose non-aug parent is in a different
  split.** This is the sibling-leak prevention.
- Under `train_val_only`, **test contains only non-aug (real) rows**. Under
  `train_only`, both val and test contain only non-aug rows.

### 1.5 KazEmo three-way split

KazEmo has only 3 speakers. Speaker-disjoint across all splits is impossible,
so the config picks two speakers for train and one for val+test, then
randomly partitions the third speaker's rows 50/50 between val and test
(seeded, reproducible). This is handled by
`splits/kazemo_speakers.py:kazemo_three_way_split`.

### 1.6 Leak-check report (`summary.json.leak_check`)

```json
{
  "passed": true,
  "per_split_aug_count":  { "train": 2105, "val": 520, "test": 0 },
  "per_corpus_aug_count": { "batch01": { "train": 2105, "val": 520, "test": 0 } },
  "policy_per_corpus":    { "batch01": ["train","val"], "batch02": [], "kazemo": [] },
  "violations":           [],
  "total_violations":     0
}
```

`passed: false` aborts the build. Re-run after inspecting `violations`.

### 1.7 Config knobs

Top-level:

- `name` — output directory name
- `seed` — deterministic splitting
- `stratify_by` — `emotion` (default)
- `speaker_disjoint: true` — forbid train↔val and train↔test speaker overlap
  (val↔test overlap allowed only for KazEmo)
- `checks.max_ratio_drift_pp` — warn if actual ratios drift > N pp from
  target (drift > threshold does **not** fail the build; a warning is printed)
- `checks.fail_on_speaker_overlap` / `fail_on_row_overlap` — abort on leaks

Per-corpus:

- `mode: split | train_only | off` — participate as a 3-way split, or only
  contribute to train, or skip entirely
- `hf_id`, `hf_split`, `cache_dir`
- `ratios: { train, val, test }`
- `speaker_column` (batch01) or `group_by: source_recording` (batch02)
- `augmented_policy` — see §1.4

---

## 2. Experiments

All four scripts live in `experiments/` and follow the same shape: a small
`run()` helper + one line per cell. Each cell gets its own `results/<label>_<ts>/`
output directory. Splits are built on demand and cached by name.

Common environment controls (set before invoking, or via `.env`):

- `CUDA_VISIBLE_DEVICES` — GPU id (default 1)
- `BACKBONE` — `hubert` or `wavlm` (each script sets its own)
- `TRAIN_CONFIG_NAME` — picks `configs/train/<BACKBONE>_<stub>.yaml`

### 2.1 Experiment 1 — HuBERT × 3 compositions, augmentation ON

File: `experiments/experiment_1.sh`  •  Backbone: HuBERT-base (95M)

| label                  | split                               | regime         |
|------------------------|-------------------------------------|----------------|
| exp1-hubert-b1         | batch01_only_aug                    | stage1 + stage2 |
| exp1-hubert-b1-b2      | batch01_plus_batch02_aug            | stage1 + stage2 |
| exp1-hubert-b1-b2-kz   | all_three_aug                       | stage1 + stage2 |

- **Stage 1** (`hubert_stage1.yaml`): backbone frozen, heads only. 20 epochs,
  LR 1e-3, cosine hold=3 / decay=17, class-balanced CE.
- **Stage 2** (`hubert_stage2.yaml`): resume from stage-1 best, unfreeze the
  contiguous top-3 encoder layers + heads, discriminative LR
  (backbone 5e-5 top, layer_decay 0.85; head 2e-4), weight decay 0.05,
  SpecAugment on, drop-path 0.15, 40 epochs, early stopping patience 6.

Purpose: measure how a two-stage partial-finetune pipeline scales with
augmentation-on data as we add corpora.

### 2.2 Experiment 2 — WavLM × 3 compositions, augmentation ON

File: `experiments/experiment_2.sh`  •  Backbone: WavLM-base-plus (95M)

Same three compositions and same stage-1/stage-2 regime as Exp 1, but using
`wavlm_stage1.yaml` / `wavlm_stage2.yaml`. Purpose: head-to-head backbone
comparison at matched data + training recipe.

### 2.3 Experiment 3 — both backbones × 3 compositions, augmentation OFF

File: `experiments/experiment_3.sh`  •  Backbones: HuBERT + WavLM (6 cells)

Uses the `*_noaug.yaml` split configs, which set `augmented_policy: drop_all`
on every corpus. Train/val/test contain non-augmented rows only.

Purpose: isolate the contribution of augmentation. Read: Exp 1 vs Exp 3 for
HuBERT, Exp 2 vs Exp 3 for WavLM.

### 2.4 Experiment 4 — full-finetune-from-epoch-0 ablation

File: `experiments/experiment_4.sh`  •  Backbones: HuBERT + WavLM (2 cells)

Configs: `hubert_full_finetune.yaml` / `wavlm_full_finetune.yaml`.

- `freeze_backbone: false` — all 12 encoder layers unfrozen from epoch 0
- Discriminative LR: backbone 1e-5 top, layer_decay 0.85; head 2e-4
- Weight decay 0.05, drop-path 0.1, SpecAugment on
- **60 epochs** (matches Exp 1/2 compute budget of stage1 20 + stage2 40)
- **Early stopping OFF** — fixed compute budget for fair comparison
- Cosine hold=3 / decay=57

Runs on `all_three_aug` (currently wired to the best Exp 1-3 composition —
edit the split argument in the script if a different composition wins).

Purpose: justify the two-stage pipeline. Compare vs Exp 1/2 on the same data:

- Two-stage ≥ full-FT → frozen-head warmup + partial unfreeze is the right
  recipe at this compute.
- Full-FT wins → drop the pipeline complexity; run full-FT directly.

---

## 3. Logging and outputs

Per cell, under `results/<label>_<ts>/`:

- `train.log` / `finetune.log` — stdout streams
- `step_train_metrics.jsonl` — **per training step** train loss (total + per
  task). Use this to rebuild training-loss curves at step granularity.
- `step_val_metrics.json` — validation metrics logged at N step intervals
- `training_metrics.json` — per-epoch record: train block, per-corpus val
  blocks, learned layer preferences. No macro aggregate stored.
- `best_model.pt` / `latest_checkpoint.pt` — best tracks the sum of
  per-corpus val totals (not stored as a labeled metric).
- `test_results.json` / `test_results_finetune.json` — per-corpus test
  metrics at the end of each stage. No `macro_total` field — read
  `per_corpus[<name>].metrics` directly.
- `finetune/…` subdir holds the stage-2 equivalents.

WandB (when enabled) mirrors the same per-step train losses and per-corpus
val/test metrics. `step=global_step` so the x-axis is training steps, not
val events.
