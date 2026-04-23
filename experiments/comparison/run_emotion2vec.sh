#!/usr/bin/env bash
# Run all emotion2vec variants (and optionally our own checkpoint) against the
# per-corpus test slices of a split dir.
#
# Usage:
#   experiments/comparison/run_emotion2vec.sh <split_dir> [max_samples] [our_checkpoint...]
#
# Examples:
#   experiments/comparison/run_emotion2vec.sh splits/b1_b2_noaug_v1
#   experiments/comparison/run_emotion2vec.sh splits/b1_b2_noaug_v1 200
#   experiments/comparison/run_emotion2vec.sh splits/b1_b2_noaug_v1 "" \
#       models/exp1-hubert-b1-b2_20260421-102709/finetune/best_model_finetuned.pt
set -euo pipefail

SPLIT_DIR=${1:?"usage: $0 <split_dir> [max_samples] [our_checkpoint...]"}
MAX_SAMPLES=${2:-}
shift $(( $# >= 2 ? 2 : $# ))
OUR_CHECKPOINTS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VARIANTS=(
  "iic/emotion2vec_base_finetuned"
  "iic/emotion2vec_plus_seed"
  "iic/emotion2vec_plus_base"
  "iic/emotion2vec_plus_large"
)

EXTRA=()
if [[ -n "$MAX_SAMPLES" ]]; then
  EXTRA+=(--max-samples "$MAX_SAMPLES")
fi

for v in "${VARIANTS[@]}"; do
  echo "=== variant: $v ==="
  python comparison/evaluate.py \
    --split-dir "$SPLIT_DIR" \
    --variant "$v" \
    "${EXTRA[@]}"
done

for ckpt in "${OUR_CHECKPOINTS[@]}"; do
  echo "=== checkpoint: $ckpt ==="
  python comparison/evaluate.py \
    --split-dir "$SPLIT_DIR" \
    --checkpoint "$ckpt" \
    "${EXTRA[@]}"
done
