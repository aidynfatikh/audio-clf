# Hugging Face Publishing Guide (3 Models, 3 Repos)

This guide explains how scripts/upload_to_huggingface.py works and how to publish three separately trained models (for three different datasets) into three separate Hugging Face model repositories.

## What The Upload Script Does

File: scripts/upload_to_huggingface.py

The script performs these steps:

1. Resolves which checkpoint file to publish.
   - Built-in choices:
     - best
     - latest
     - interrupted
     - best-finetuned
     - latest-finetuned
     - path (explicit custom path)

2. Creates (or reuses) the target Hugging Face model repo.
   - Supports user repo or organization repo.

3. Builds a temporary export package with:
   - Selected checkpoint file (.pt)
   - label_encoders.json (from --label-encoders-path)
   - Optional metrics JSON (auto or --metrics-path)
   - export_config.json (generated metadata)
   - README.md model card (generated)

4. Uploads the package to the selected repo and revision.

## Required Setup

1. Install dependencies:

   pip install -r requirements.txt

2. Authenticate with Hugging Face token:

   export HF_TOKEN=your_hf_token

3. Confirm script help:

   python scripts/upload_to_huggingface.py --help

## Important Arguments

- --repo-name
  - Destination model repository name.

- --org-name
  - Optional organization namespace.
  - If omitted, uploads to your user account.

- --checkpoint
  - Select source checkpoint by alias, or use path.

- --checkpoint-path
  - Required when --checkpoint path.

- --label-encoders-path
  - Path to the correct label_encoders.json for that specific model.
  - This is important when different datasets produce different label maps.

- --include-metrics
  - Include metrics JSON in uploaded package.

- --model-card-path
  - Optional path to a custom model card template that will be uploaded as README.md.
  - Supported placeholders in template content:
    - `{{REPO_ID}}`
    - `{{CHECKPOINT_NAME}}`
    - `{{INCLUDE_METRICS_LINE}}`

- --metrics-path
  - Optional explicit path to metrics JSON.
  - Useful when each model has its own metrics file location.

- --private
  - Create/private repo.

- --revision
  - Branch/revision target (default: main).

## Publish 3 Models To 3 Repos (Example)

Assume you have three trained models from three datasets:

- Dataset A model checkpoint: models/best_model_dataset_a.pt
- Dataset B model checkpoint: models/best_model_dataset_b.pt
- Dataset C model checkpoint: models-s/best_model.pt

And matching label encoders:

- Dataset A labels: models/label_encoders_dataset_a.json
- Dataset B labels: models/label_encoders_dataset_b.json
- Dataset C labels: models-s/label_encoders.json

And optional metrics:

- Dataset A metrics: models/training_metrics_dataset_a.json
- Dataset B metrics: models/training_metrics_dataset_b.json
- Dataset C metrics: models-s/training_metrics.json

### Repo 1 (Dataset A)

python scripts/upload_to_huggingface.py \
  --repo-name audio-clf-dataset-a \
  --org-name aidynfatikh \
  --checkpoint path \
  --checkpoint-path models/best_model_dataset_a.pt \
  --label-encoders-path models/label_encoders_dataset_a.json \
  --include-metrics \
  --metrics-path models/training_metrics_dataset_a.json

### Repo 2 (Dataset B)

python scripts/upload_to_huggingface.py \
  --repo-name audio-clf-dataset-b \
  --org-name aidynfatikh \
  --checkpoint path \
  --checkpoint-path models/best_model_dataset_b.pt \
  --label-encoders-path models/label_encoders_dataset_b.json \
  --include-metrics \
  --metrics-path models/training_metrics_dataset_b.json

### Repo 3 (Dataset C)

python scripts/upload_to_huggingface.py \
  --repo-name audio-clf-dataset-c \
  --org-name aidynfatikh \
  --checkpoint path \
  --checkpoint-path models-s/best_model.pt \
  --label-encoders-path models-s/label_encoders.json \
  --include-metrics \
  --metrics-path models-s/training_metrics.json

## Verify After Upload

For each repository:

1. Open the repo URL printed by the script.
2. Confirm files exist:
   - checkpoint .pt
   - label_encoders.json
   - export_config.json
   - README.md
   - metrics JSON (if requested)
3. Confirm README describes the intended checkpoint.
4. Confirm label names match your dataset classes.

## Recommended Naming Convention

Use stable, dataset-specific names to avoid confusion:

- audio-clf-dataset-a
- audio-clf-dataset-b
- audio-clf-dataset-c

Optional tags for discoverability:

--tags "audio-classification,multi-task,hubert,dataset-a"

## Custom Model Card

Create a template file, for example `docs/model_card_template.md`:

There is now a ready-to-use template in this repo at:
- `docs/model_card_template.md`

---
language:
- en
library_name: pytorch
pipeline_tag: audio-classification
---

# {{REPO_ID}}

This model was published from checkpoint `{{CHECKPOINT_NAME}}`.

{{INCLUDE_METRICS_LINE}}

Then push with:

python scripts/upload_to_huggingface.py \
  --repo-name audio-clf-dataset-a \
  --org-name aidynfatikh \
  --checkpoint path \
  --checkpoint-path models/best_model_dataset_a.pt \
  --label-encoders-path models/label_encoders_dataset_a.json \
  --include-metrics \
  --metrics-path models/training_metrics_dataset_a.json \
  --model-card-path docs/model_card_template.md

## Troubleshooting

- Error: HF token missing
  - Set HF_TOKEN or pass --hf-token.

- Error: checkpoint not found
  - Check relative path is from repository root.

- Wrong labels in repo
  - Ensure --label-encoders-path points to the correct file for that model.

- Metrics missing
  - Use --include-metrics and provide --metrics-path explicitly when auto-detection does not match your layout.
