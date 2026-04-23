#!/usr/bin/env bash
# Run all emotion2vec variants (and optionally our own checkpoints) against a
# single HuggingFace dataset (one corpus, e.g. Turkish emotion audio).
#
# Usage:
#   experiments/comparison/run_hf_dataset.sh <hf_dataset_id> <max_samples> [checkpoint...]
#
# Examples:
#   experiments/comparison/run_hf_dataset.sh umutkkgz/tr-full-dataset 500
#   experiments/comparison/run_hf_dataset.sh umutkkgz/tr-full-dataset 500 \
#       models/exp1-hubert-b1-b2_20260421-102709/finetune/best_model_finetuned.pt
set -euo pipefail

HF_DATASET=${1:?"usage: $0 <hf_dataset_id> <max_samples> [checkpoint...]"}
MAX_SAMPLES=${2:?"max_samples required (HF datasets can be huge; don't pull it all)"}
shift 2
OUR_CHECKPOINTS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VARIANTS=(
  "iic/emotion2vec_base_finetuned"
  "iic/emotion2vec_plus_seed"
  "iic/emotion2vec_plus_base"
  "iic/emotion2vec_plus_large"
)
HUB="${E2V_HUB:-ms}"

for v in "${VARIANTS[@]}"; do
  echo "=== variant: $v (hub=$HUB) ==="
  python comparison/evaluate.py \
    --hf-dataset "$HF_DATASET" \
    --max-samples "$MAX_SAMPLES" \
    --variant "$v" \
    --hub "$HUB"
done

for ckpt in "${OUR_CHECKPOINTS[@]}"; do
  echo "=== checkpoint: $ckpt ==="
  python comparison/evaluate.py \
    --hf-dataset "$HF_DATASET" \
    --max-samples "$MAX_SAMPLES" \
    --checkpoint "$ckpt"
done
