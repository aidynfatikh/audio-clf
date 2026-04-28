#!/usr/bin/env bash
# Run the full comparison matrix (emotion2vec variants + Gemini + Qwen +
# optional local checkpoints) against a single HuggingFace dataset (one
# corpus, e.g. Turkish emotion audio). Each (model, dataset) pair lands in
# results/comparison/<dataset>/<model_tag>/ — a failure in one pair does not
# abort the rest.
#
# Usage:
#   experiments/comparison/run_hf_dataset.sh <hf_dataset_id> <max_samples> [checkpoint...]
#
# Examples:
#   experiments/comparison/run_hf_dataset.sh umutkkgz/tr-full-dataset 500
#   experiments/comparison/run_hf_dataset.sh umutkkgz/tr-full-dataset 500 \
#       models/exp1-hubert-b1-b2_20260421-102709/finetune/best_model_finetuned.pt
#
# Env passthrough:
#   E2V_HUB                emotion2vec hub (ms|hf, default ms)
#   GEMINI_MODEL           gemini model id (default gemini-3.1-pro-preview)
#   GEMINI_CONCURRENCY     parallel API calls (default 4)
#   GEMINI_API_KEY         required for gemini run; loaded from .env if unset
#   QWEN_MODEL             qwen2-audio HF id (default Qwen/Qwen2-Audio-7B-Instruct)
#   SKIP_GEMINI=1          skip the gemini leg
#   SKIP_QWEN=1            skip the qwen leg
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

run_or_warn() {
  local label="$1"; shift
  echo "────────────────────────────────────────────────────────────────"
  echo ">>> $label"
  echo "────────────────────────────────────────────────────────────────"
  if ! "$@"; then
    echo "!!! $label FAILED — continuing with the rest of the matrix." >&2
  fi
}

# 1) emotion2vec variants
for v in "${VARIANTS[@]}"; do
  run_or_warn "emotion2vec=$v on $HF_DATASET" \
    python comparison/evaluate.py \
      --hf-dataset "$HF_DATASET" \
      --max-samples "$MAX_SAMPLES" \
      --variant "$v" \
      --hub "$HUB"
done

# 2) Gemini API
if [[ "${SKIP_GEMINI:-}" != "1" ]]; then
  run_or_warn "gemini on $HF_DATASET" \
    experiments/comparison/run_gemini.sh \
      --hf-dataset "$HF_DATASET" "$MAX_SAMPLES"
fi

# 3) Qwen2-Audio (local)
if [[ "${SKIP_QWEN:-}" != "1" ]]; then
  run_or_warn "qwen on $HF_DATASET" \
    experiments/comparison/run_qwen.sh \
      --hf-dataset "$HF_DATASET" "$MAX_SAMPLES"
fi

# 4) Our local checkpoints
for ckpt in "${OUR_CHECKPOINTS[@]}"; do
  run_or_warn "checkpoint=$ckpt on $HF_DATASET" \
    python comparison/evaluate.py \
      --hf-dataset "$HF_DATASET" \
      --max-samples "$MAX_SAMPLES" \
      --checkpoint "$ckpt"
done
