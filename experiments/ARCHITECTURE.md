# Architecture & training-recipe debrief

What we're running, where each hyperparameter comes from. Every non-obvious
choice has a citation. Bib entries live in `paper/refs_partial_finetune.bib`.

## 1. Model (`multihead/model.py`)

### Backbone

Two interchangeable SSL encoders, 95M params each, 12 transformer layers,
768-d hidden:

- **HuBERT-base** — `facebook/hubert-base-ls960`
- **WavLM-base-plus** — `microsoft/wavlm-base-plus`

Selected by config at `model.name` / `model.pretrained`. Swap is a one-line
change in the YAML; no Python edits needed.

Source: both are the de-facto bases for SSL speech. Wang et al. 2021 benchmark
them head-to-head for SER on IEMOCAP and both sit near the top.
→ [wang2021ssl_benchmark](https://arxiv.org/abs/2111.02735)

### Multi-task heads

Three parallel MLP heads (emotion, gender, age). Each head is
`Linear(768→256) → ReLU → Dropout → Linear(256→K)`. Dropout 0.2 on the
emotion head, 0.1 on gender/age (emotion is the hardest and most
overfitting-prone task in the data).

Multi-task SER with shared SSL encoder + task-specific heads is standard.
→ [amazon_mt_ser_2022](https://assets.amazon.science/22/85/2c46e16c457caec87a82ca3363eb/multi-lingual-multi-task-speech-emotion-recognition-using-wav2vec-2.0.pdf),
[pepino2021wav2vec_ser](https://arxiv.org/abs/2104.03502),
[burkhardt2022_age_ssl](http://www.apsipa.org/proceedings/2022/APSIPA%202022/ThPM2-3/1570834111.pdf)

### Per-task weighted layer sum

Each head gets its own **learnable softmax over all 13 hidden states** (12
transformer layers + 1 feature-projection embedding), then mean-pools over
time. The weights are stored as `nn.Parameter(torch.ones(13))` per task;
softmax is applied in the forward pass. This lets each task choose its own
depth: emotion typically pulls higher layers, gender lower, consistent with
layer-wise probing studies.

Source (weighted-sum pooling from SSL layers + per-task preference):
→ [pepino2021wav2vec_ser](https://arxiv.org/abs/2104.03502) (wav2vec2 weighted
sum for SER), [ssl_layerwise_naacl2025](https://aclanthology.org/2025.findings-naacl.227.pdf)
(layer-wise analysis showing task-specific optima differ across depth).

### Gender-conditioned SER context

GEmo-CLAP motivates the joint emotion+gender setup — gender is a strong
conditioning signal for emotion.
→ [gemoclap2023](https://arxiv.org/pdf/2306.07848)

## 2. Data pipeline

### Splits (`splits/…`)

- Stratified grouped 3-way split: stratify by emotion, group by speaker
  (batch01) or source recording (batch02) to guarantee speaker-disjoint
  train/val/test.
- KazEmo has only 3 speakers, so 2 go to train and the 3rd is split 50/50
  between val and test — this is the smallest possible leakage-free setup
  when speaker-count is bounded.
- Augmentation-aware routing via `augmented_policy` (drop_all / train_only /
  train_val_only / keep_all). Sibling-leak prevention is enforced in code
  via `_compute_leak_check` — build aborts on any violation.

Full details in `experiments/README.md` §1.

### Runtime augmentation (`utils/data.py`)

Applied on-the-fly to **training** batches only:

- **Speed perturbation** at 0.9 / 1.0 / 1.1, 80 % probability
- **Noise mixing** at 5–20 dB SNR, 50 % probability, only if `noise_dir` set

Standard SpeechBrain-style augmentation stack.
→ [speechbrain2021](https://speechbrain.readthedocs.io/en/latest/tutorials/nn/using-wav2vec-2.0-hubert-wavlm-and-whisper-from-huggingface-with-speechbrain.html)

### SpecAugment

Enabled on stage-2 and full-finetune only (not stage-1, where the backbone
is frozen and masking noise on a frozen feature extractor doesn't help).
Uses the model's built-in pre-transformer masking:
`mask_time_prob=0.05`, `mask_time_length=10`, `mask_feature_prob=0.0`.

Source: SpecAugment as a proven regularizer for fine-tuned SSL encoders.
→ [pepino2021finetune_ser](https://arxiv.org/abs/2110.06309) shows
SpecAugment helps wav2vec2 SER fine-tuning at these magnitudes.

## 3. Training regime

### Stage 1 — frozen backbone, heads only

Config: `configs/train/{hubert,wavlm}_stage1.yaml`.

| knob                   | value             |
|------------------------|-------------------|
| freeze_backbone        | true              |
| num_epochs             | 20                |
| head_learning_rate     | 1e-3              |
| weight_decay           | 0.01              |
| label_smoothing        | 0.1               |
| class_weighting        | balanced (sklearn)|
| scheduler              | cosine, hold 3 / decay 17 |
| SpecAugment            | off               |

Classic linear-probe-first step: get the heads to a reasonable operating
point before touching backbone weights. Keeps stage-2 starting from a good
initialization and avoids early-backbone corruption.
→ [howard2018ulmfit](https://arxiv.org/abs/1801.06146) (gradual unfreezing)

### Stage 2 — partial finetune with discriminative LR

Config: `configs/train/{hubert,wavlm}_stage2.yaml`.

| knob                    | value            |
|-------------------------|------------------|
| freeze_backbone (init)  | true, then unfreeze top-N |
| unfreeze_top_n          | 3                |
| unfreeze_contiguous     | true             |
| layer_decay             | 0.85             |
| backbone_lr_top         | 5e-5             |
| head_lr                 | 2e-4             |
| training_drop_path      | 0.15             |
| weight_decay            | 0.05             |
| SpecAugment             | on               |
| num_epochs              | 40               |
| early_stopping_patience | 6                |
| scheduler               | cosine, hold 5 / decay 35, scale_group_decay |

**Top-N contiguous unfreeze.** We use the learned per-task layer preferences
from stage 1 to score each encoder layer, then pick the **contiguous window
of 3** with the highest summed importance (`select_contiguous_top_n` in
`utils/finetune_utils.py`). Contiguity keeps gradient flow unbroken — a
non-contiguous choice like {layer 2, 8, 11} forces gradients to flow through
a frozen layer 3–7 band that can't co-adapt.

Source: partial top-N unfreezing with contiguous windows is the recipe used
by Pepino and followed by most downstream SER/SLU work.
→ [pepino2021finetune_ser](https://arxiv.org/abs/2110.06309),
[wang2021ssl_benchmark](https://arxiv.org/abs/2111.02735) §4

**Discriminative LR decay (0.85).** Each unfrozen layer *i* gets
`backbone_lr_top * 0.85^(n_layers - i)`. Shallower layers → smaller LR. Heads
use `head_lr = 2e-4`. This is ULMFiT's discriminative fine-tuning rate
schedule adapted to the transformer-layer count.
→ [howard2018ulmfit](https://arxiv.org/abs/1801.06146), also exposed
directly in SpeechBrain wav2vec2 recipes
→ [speechbrain2021](https://speechbrain.readthedocs.io/en/latest/tutorials/nn/using-wav2vec-2.0-hubert-wavlm-and-whisper-from-huggingface-with-speechbrain.html)

**Scheduler `scale_group_decay: true`.** Higher-LR groups (head) decay more
aggressively than lower-LR groups (bottom backbone layers) during cosine
annealing — a small correction that keeps shallow layers useful for
longer. `group_power_lo=0.8 / hi=1.3` controls the aggressiveness spread.

**Regularization stack.** `training_drop_path: 0.15` + `weight_decay: 0.05`
+ label smoothing 0.1 + class-balanced CE. Stronger than stage-1 because the
backbone is now trainable and the effective-parameter count jumped from
~1 M (heads) to ~25 M (3 layers × ~7 M + heads). At ~4–8 k training rows,
this is the regularization level that keeps us on the safe side of the
overfitting cliff.
→ [exhubert2024](https://arxiv.org/html/2406.10275v1) §3 uses drop-path
0.1–0.2 and weight-decay 0.01–0.1 for HuBERT-based SER fine-tunes;
→ [pepino2021finetune_ser](https://arxiv.org/abs/2110.06309) reports
label-smoothing 0.1 as a standard choice in this setting.

### Full-finetune from epoch 0 (Exp 4 only)

Config: `configs/train/{hubert,wavlm}_full_finetune.yaml`.

All 12 encoder layers unfrozen from step 1. Same discriminative LR recipe
with a **smaller top LR (1e-5)** to compensate for more trainable params.
60 epochs (= stage1 20 + stage2 40) for compute parity with Exp 1/2, early
stopping off so the budget is fixed.

Purpose: ablation justifying the two-stage pipeline. The freeze-then-unfreeze
recipe (ULMFiT-style) is reported to match or beat end-to-end fine-tuning at
fixed compute in low-data regimes; Exp 4 tests that claim on our data.
→ [howard2018ulmfit](https://arxiv.org/abs/1801.06146) §3.3

### Class-balanced cross-entropy

`class_weighting: balanced` uses sklearn's `compute_class_weight`:
`w_i = N / (K * n_i)`. Prevents the common gender / rare emotion (e.g. fear,
disgust) from being drowned out. Only the emotion head sees real imbalance
in practice, but applying it uniformly is cheaper than per-task tuning.

### Optimizer

Fused AdamW on CUDA when available. β defaults (0.9 / 0.999). One param
group per unfrozen layer + one for the heads, each with its own LR.
`grad_clip_norm: 1.0` globally.

### Mixed precision

`torch.amp.autocast` with bf16 on A100-class GPUs (we detect
`torch.cuda.is_bf16_supported()`). fp32 master weights. No GradScaler needed
for bf16.

## 4. What we did **not** tune

These are held fixed at literature defaults. Each is a candidate for a
separate focused ablation, but they are **not** what Exp 1–4 are measuring:

- `layer_decay = 0.85` (ULMFiT range 0.80–0.95)
- `unfreeze_top_n = 3` (Pepino uses 2–4; we picked mid)
- `training_drop_path = 0.15` (ExHuBERT range 0.1–0.2)
- `weight_decay = 0.05` stage-2 / `0.01` stage-1 (standard AdamW defaults)
- `label_smoothing = 0.1` (standard)
- Head architecture (single hidden layer, 256-d, dropout 0.1–0.2)
- Runtime augmentation magnitudes (speed 0.9–1.1, SNR 5–20 dB)
- SpecAugment `mask_time_prob=0.05, mask_time_length=10` (HF defaults)

If any of these becomes the subject of a reviewer question, the answer is:
*"held at literature default, not the question this experiment answers —
recipe-sensitivity ablation is future work."*

## 5. What Exp 1–4 actually test

- **Exp 1 vs Exp 2**: HuBERT vs WavLM at matched recipe + data.
- **Exp 1 vs Exp 3** (and Exp 2 vs Exp 3): augmentation-on vs augmentation-off.
- **Intra-experiment (b1 / b1+b2 / b1+b2+kz)**: does more data help?
- **Exp 4 vs Exp 1/2 on the best composition**: is the two-stage pipeline
  actually better than end-to-end full finetune at matched compute?

Everything else — architecture, optimizer, regularization, augmentation
magnitudes — is held constant so these four comparisons are clean.

## 6. Pointer to code

- Model: `multihead/model.py` (`MultiTaskBackbone`)
- Layer-importance + contiguous top-N picker: `utils/finetune_utils.py`
- Discriminative optimizer builder: `utils/finetune_utils.py:build_optimizer`
- Cosine schedule with per-group decay scaling: `utils/checkpointing.py:make_cosine_schedule`
- Stage 1 training loop: `multihead/train.py`
- Stage 2 training loop: `multihead/finetune.py`
- Dataset + runtime augmentation: `utils/data.py`
- Split pipeline: `splits/builder.py`, `splits/sources.py`
- Bib: `paper/refs_partial_finetune.bib`
