# Dataset splits — build, layout, and leakage guarantees

This repo trains on three corpora: **batch01**, **batch02**, and **KazEmoTTS**.
Each source has different speaker coverage, label completeness, and aug-vs-clean
composition, so splits are built from a single YAML config and materialized as
parquet manifests under `splits/<name>/`. The training code then rehydrates
those manifests back into HuggingFace `Dataset` objects.

This doc covers:

1. The three data modes the training loader supports, and which one to use.
2. What the split-builder YAML looks like and what each knob controls.
3. How train / val / test are chosen and how speaker leakage is prevented.
4. What happens to augmented rows.
5. The consistency checks that run at build time.

---

## 1. Three ways the training loader can get a train/val/test

`utils/data_loading.py::build_mixed_train_val_splits()` dispatches by env var:

| Env var              | Mode                 | Speaker-disjoint? | Recommended? |
|----------------------|----------------------|-------------------|--------------|
| `SPLIT_MANIFEST_DIR` | Manifest mode        | **Yes**           | **Yes — default for all real runs.** |
| `TRAIN_VAL_MANIFEST` | Legacy holdout mode  | No (row-id only)  | Only for reproducing old validate.py runs. |
| (neither set)        | HF-split fallback    | No                | Smoke tests only. |

Only the **manifest mode** builds splits the way this doc describes. The other
two paths predate the builder and split by row_id — a single speaker can land in
both train and val. Set `SPLIT_MANIFEST_DIR=splits/batch01_only_v1` (or the
split directory you materialized) for anything you intend to publish.

---

## 2. Split-builder YAML — what each field does

Example (`splits/batch01_only_v1/config.yaml`):

```yaml
name: batch01_only_v1           # output dir will be splits/<name>/
seed: 42                        # drives every random choice
speaker_disjoint: true          # group rows by speaker_id when splitting
stratify_by: emotion            # preserve emotion distribution across splits
output_dir: splits              # parent dir for the materialized split

corpora:
  batch01:
    mode: split                 # split | train_only | off
    hf_id: 01gumano1d/batch01-validation-test
    hf_split: train
    cache_dir: data/batch01-validation-test
    ratios: {train: 0.70, val: 0.15, test: 0.15}
    filter_augmented: true      # legacy alias for augmented_policy: drop_all
    speaker_column: speaker_id

  batch02:
    mode: "off"                 # YAML 1.1 parses bare 'off' as False; quote it
  kazemo:
    mode: "off"

checks:
  fail_on_speaker_overlap: true
  fail_on_row_overlap: true
  max_ratio_drift_pp: 5
  min_samples_per_emotion_per_split: 10
```

### Top-level keys

- **`name`** — becomes `splits/<name>/`. Must be unique; the builder refuses to
  overwrite a non-empty existing directory unless called with `force=True`.
- **`seed`** — single RNG seed for every corpus-level decision. Reuse the same
  `name` + `seed` to regenerate the identical manifests.
- **`speaker_disjoint`** — when `true` (the default), rows are grouped by
  `speaker_id` before splitting so a speaker cannot cross train/val/test. When
  `false`, rows are treated as independent (group = row_id).
- **`stratify_by`** — `emotion` or `emotion+gender+age`. Drives the stratified
  fold selection so each split carries a comparable label distribution.
- **`output_dir`** — parent of the per-split directory; relative paths resolve
  from the repo root.

### Per-corpus keys (`corpora.<name>`)

- **`mode`**:
  - `split` — full 3-way split using the corpus's `ratios`.
  - `train_only` — every row goes to train (`val` and `test` stay empty for
    this corpus). Useful for extra-pretraining data with no held-out set.
  - `off` — corpus is ignored entirely. **Always quote `"off"`** — unquoted
    `off` is parsed as the boolean `False` by YAML 1.1, but the builder's
    `_normalize_mode` accepts both.
- **`hf_id` / `hf_split` / `cache_dir`** — forwarded to `datasets.load_dataset`.
- **`ratios`** — `{train, val, test}` fractions. Used by stratified-grouped
  splitting (batch01/batch02 only). Ignored by KazEmo's custom path.
- **`stratify_by`** — per-corpus override of the top-level `stratify_by`.
- **`speaker_column`** — name of the source column to read `speaker_id` from.
  If omitted, rows carry `speaker_id = None` and fall back to per-row groups
  (see "Group fallback" below).
- **`augmented_policy`** — `drop_all` | `train_only` | `keep_all`. See §4.
- **`filter_augmented`** — legacy alias (`true` → `drop_all`, `false` →
  `keep_all`). Still supported for old configs.

### KazEmo-specific keys

KazEmoTTS has only 3 speakers, so it uses a fixed assignment instead of
stratified folds:

```yaml
kazemo:
  mode: split
  speaker_selection:
    strategy: by_index         # or by_name
    train_indices: [0, 1]
    valtest_index: 2
    # or: train_names: [...], valtest_name: "..."
  valtest_ratio: {val: 0.5, test: 0.5}
  max_samples: 20000
```

- Train speakers never appear in val/test.
- The one remaining speaker's rows are shuffled (seeded) and split into val
  and test per `valtest_ratio`. **This is the one place we intentionally allow
  a single speaker to appear in both val and test** — otherwise with 3 speakers
  we could not have both splits.

### `checks` block

Every key is optional; defaults shown:

- `fail_on_speaker_overlap: true` — raise after writing if any speaker leaks.
- `fail_on_row_overlap: true` — raise if the same `(dataset, row_id)` lands in
  two splits.
- `max_ratio_drift_pp: 5` — for `mode: split` corpora, warn if the actual
  per-split share drifts more than N percentage points from the configured
  ratio.
- `min_samples_per_emotion_per_split: 0` — warn when any emotion has fewer
  than this many rows in a split.

---

## 3. How each split is chosen

### 3.1 batch01 / batch02 — stratified, grouped 3-way

Implemented in `splits/stratified_group.py::stratified_grouped_three_way`.

```
y      = emotion label per row            (or emotion|gender|age)
groups = speaker_id per row               (or unique per-row id if missing)

pass 1 (peel off test):
  k1 = round(1 / test_ratio)
  folds = StratifiedGroupKFold(n_splits=k1, shuffle=True, random_state=seed)
  test_idx = fold whose |size - test_ratio*n| is minimal

pass 2 (peel off val on remaining rows):
  remaining = rows \ test_idx
  k2 = round(1 / (val_ratio / (1 - test_ratio)))
  folds = StratifiedGroupKFold(n_splits=k2, shuffle=True, random_state=seed+1)
  val_idx = fold (in remaining) closest to target

train_idx = remaining \ val_idx
```

Why this is speaker-disjoint **by construction**:

- `StratifiedGroupKFold` guarantees every group lands in exactly one fold per
  call. Pick any single fold → all other folds have different groups.
- Pass 2 operates only on `remaining = rows \ test_idx`, so any group that
  appeared in test is physically absent from the val pool. Train and val
  cannot see a test speaker, and val cannot see a train speaker either.

### 3.2 KazEmo — speaker-partitioned 3-way

Implemented in `splits/kazemo_speakers.py::kazemo_three_way_split`.

1. `resolve_kazemo_speakers` picks the train speaker set and the one valtest
   speaker (by index or by name). It raises if the valtest speaker overlaps
   train.
2. Rows from train speakers → train.
3. Rows from the valtest speaker → shuffled with the run seed and partitioned
   into val / test according to `valtest_ratio`.
4. Rows from any other (unassigned) speaker are silently skipped.

Speaker parsing lives in `splits/kazemo_speakers.py::parse_kazemo_speaker` —
it walks filename/text tokens, stopping at the first emotion token from
`loaders.kazemo.load_data._EMO_MAP`, and returns everything before it as the
speaker id.

### 3.3 Group fallback — rows without a speaker

If `speaker_column` is missing (or the value is null/empty), the row gets a
synthetic unique group key `__singleton__<index>`. Each such row is its own
group, which degenerates to plain stratified splitting for those rows. This
means: **no speaker info = no cross-split guarantee for that row** — you are
trusting the individual row to be safe.

---

## 4. Augmented rows

`splits/augmented_filter.py::extract_augmented_flag` reads the flag from the
top level (`augmented` / `is_augmented`) or from a nested `metadata` dict or
JSON string. Returns `True`, `False`, or `None` (unknown).

The builder picks one of three policies per corpus:

| Policy       | At load time               | At split time                                         |
|--------------|----------------------------|-------------------------------------------------------|
| `drop_all`   | Aug rows filtered out      | Split runs only on clean rows.                        |
| `keep_all`   | Aug rows kept              | Every row (aug or clean) is 3-way split normally.     |
| `train_only` | Aug rows kept              | Clean rows are 3-way split; aug rows go to train only, **unless** their speaker landed in val/test, in which case they're dropped. Rows with unknown speaker are also dropped (can't prove no leakage). |

**Default (when neither `augmented_policy` nor `filter_augmented` is set):**

- `speaker_column` is configured → `train_only`. We can prove aug routing
  doesn't leak.
- No `speaker_column` → `drop_all`. Without speaker attribution there's no way
  to confirm an augmented copy of a held-out utterance isn't sneaking into
  train, so we refuse to take the risk.

Pick `drop_all` when you want a clean eval on the held-out set and don't need
the extra aug rows in train. Pick `train_only` when you do want them but you
trust the augmentation pipeline's speaker attribution. Pick `keep_all` only
for ablation experiments where you specifically want to measure the effect of
augmented rows in val/test.

---

## 5. Consistency checks run at build time

After assignments are chosen and before parquet files are written:

1. **Speaker leakage** (`_check_speaker_disjoint`, per dataset):
   - train ↔ val and train ↔ test are **never** allowed.
   - val ↔ test is allowed only for KazEmo (3-speaker corpus).
   - Leaked entries are reported as `"<dataset>:<speaker>"`. With
     `fail_on_speaker_overlap: true` (the default) any leak aborts the build
     after it prints the list.

2. **Row-id disjointness** (`_check_row_id_disjoint`):
   - Fails if the same `(dataset, row_id)` pair appears in two splits. This
     is the last line of defence; it also catches aug-vs-clean id collisions.

3. **Ratio drift** (`_check_ratio_drift`):
   - For each `mode: split` corpus, compares actual per-split share against
     the configured `ratios`. Reports any corpus/split whose drift exceeds
     `max_ratio_drift_pp`. Warn-only.

4. **Emotion coverage** (`_check_emotion_coverage`):
   - Warns when any emotion has fewer than `min_samples_per_emotion_per_split`
     rows in a split. Warn-only (small emotions in small corpora can legitimately
     undershoot).

---

## 6. What the builder writes

Under `splits/<name>/`:

```
config.yaml          — verbatim copy of the input config for provenance
summary.json         — per-split totals, per-emotion counts, speakers per split,
                       kazemo speaker assignment, ratio drift, coverage warnings
train.parquet        — one row per manifest entry (columns from MANIFEST_COLUMNS)
val.parquet
test.parquet
```

Manifest columns:

```
dataset, row_id, source_index, speaker_id, emotion, gender, age_category,
augmented, split
```

Only metadata is stored — no audio bytes. Audio is re-fetched from the HF
cache at training time.

---

## 7. How training consumes the manifests

`splits/materialize.py::materialize_split` rehydrates parquet manifests back
into HuggingFace `Dataset`s:

1. Group manifest rows by `(split, dataset)` into lists of `source_index`.
2. For each dataset used, re-open the underlying HF source (batch01/batch02
   via `load_dataset`, KazEmo via `load_kazemotts`).
3. `split.select(source_indices)` extracts exactly the rows the manifest
   pointed to — no re-sampling, no RNG.
4. Cast all splits to the canonical schema
   `{audio, emotion, gender, age_category}` so they can be concatenated.

`materialize_named_val` additionally produces the named val splits the
training loop uses for task-masked evaluation:

- `val` — concatenation of batch01 and batch02 val rows; evaluated on all
  three tasks (emotion/gender/age).
- `kazemo` — kazemo val rows only; evaluated on emotion only
  (`_KAZEMO_TASKS` restricts the metric set).

`utils/data_loading.py::build_splits_from_manifest_dir` wraps both and returns
the 5-tuple `(hf_dataset, train, val, named_val_splits, composition)` the
training scripts expect.

---

## 8. Running the builder

```bash
python -c "
from splits.builder import build_splits
build_splits('splits/batch01_only_v1/config.yaml', force=True)
"
```

Or point the training loop at an existing split directory:

```bash
SPLIT_MANIFEST_DIR=splits/batch01_only_v1 \
BACKBONE=hubert MODEL_DIR=models/hb-bo1 \
python multihead/train.py
```

---

## 9. Leakage guarantees — summary

When you use `SPLIT_MANIFEST_DIR` + a config with `speaker_disjoint: true`:

- **Train ↔ val:** speakers cannot overlap (enforced by `StratifiedGroupKFold`
  for batch01/batch02; by explicit speaker partitioning for KazEmo).
- **Train ↔ test:** same guarantee, same mechanism.
- **Val ↔ test:** disjoint for batch01/batch02. Intentionally shares one
  speaker for KazEmo; the checker exempts only KazEmo, so any regression in
  the other corpora is caught.
- **Row-id collisions:** `_check_row_id_disjoint` aborts the build if any
  `(dataset, row_id)` lands in two splits.
- **Augmented copies:** aug rows either are dropped at source (`drop_all`),
  or are forwarded only to train under a speaker-provable policy
  (`train_only`). Rows without speaker attribution are dropped under
  `train_only` to avoid silent leakage.

When you use `TRAIN_VAL_MANIFEST` or the plain HF-split fallback, none of the
above guarantees hold — only row-id exclusion. Use manifest mode.
