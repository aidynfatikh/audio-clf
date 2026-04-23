#!/usr/bin/env bash
# Run all emotion2vec variants (and optionally our own checkpoints) against the
# per-corpus test slices of a split dir.
#
# Usage:
#   experiments/comparison/run_emotion2vec.sh <split_dir> [args...]
#
# Any numeric positional arg becomes --max-samples; any other arg is treated as
# a local checkpoint path (evaluated alongside the emotion2vec variants).
#
# Examples:
#   experiments/comparison/run_emotion2vec.sh splits/b1_b2_noaug_v1
#   experiments/comparison/run_emotion2vec.sh splits/b1_b2_noaug_v1 200
#   experiments/comparison/run_emotion2vec.sh splits/b1_b2_noaug_v1 \
#       models/exp1-hubert-b1-b2_20260421-102709/finetune/best_model_finetuned.pt \
#       models/exp2-wavlm-b1-b2_20260421-180341/finetune/best_model_finetuned.pt
set -euo pipefail

SPLIT_DIR=${1:?"usage: $0 <split_dir> [max_samples|checkpoint ...]"}
shift

MAX_SAMPLES=""
OUR_CHECKPOINTS=()
for arg in "$@"; do
  if [[ "$arg" =~ ^[0-9]+$ ]]; then
    MAX_SAMPLES="$arg"
  else
    OUR_CHECKPOINTS+=("$arg")
  fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Use ModelScope names (iic/...) — FunASR's model registry is keyed on these.
# emotion2vec_base_finetuned only exists on ModelScope; plus_* exist on both.
VARIANTS=(
  "iic/emotion2vec_base_finetuned"
  "iic/emotion2vec_plus_seed"
  "iic/emotion2vec_plus_base"
  "iic/emotion2vec_plus_large"
)
HUB="${E2V_HUB:-ms}"

EXTRA=()
if [[ -n "$MAX_SAMPLES" ]]; then
  EXTRA+=(--max-samples "$MAX_SAMPLES")
fi

for v in "${VARIANTS[@]}"; do
  echo "=== variant: $v (hub=$HUB) ==="
  python comparison/evaluate.py \
    --split-dir "$SPLIT_DIR" \
    --variant "$v" \
    --hub "$HUB" \
    "${EXTRA[@]}"
done

for ckpt in "${OUR_CHECKPOINTS[@]}"; do
  echo "=== checkpoint: $ckpt ==="
  python comparison/evaluate.py \
    --split-dir "$SPLIT_DIR" \
    --checkpoint "$ckpt" \
    "${EXTRA[@]}"
done
