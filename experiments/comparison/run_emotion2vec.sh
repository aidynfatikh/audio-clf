#!/usr/bin/env bash
# Run all emotion2vec variants against the per-corpus test slices of a split dir.
#
# Usage:
#   experiments/comparison/run_emotion2vec.sh <split_dir> [max_samples]
#
# Example:
#   experiments/comparison/run_emotion2vec.sh splits/b1_b2_noaug_v1
#   experiments/comparison/run_emotion2vec.sh splits/all_three_noaug_v1 200
set -euo pipefail

SPLIT_DIR=${1:?"usage: $0 <split_dir> [max_samples]"}
MAX_SAMPLES=${2:-}

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
  echo "=== $v ==="
  python comparison/evaluate.py \
    --split-dir "$SPLIT_DIR" \
    --variant "$v" \
    "${EXTRA[@]}"
done
