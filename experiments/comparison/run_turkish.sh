#!/usr/bin/env bash
# Run the comparison harness against both public Turkish emotion datasets on
# HuggingFace. Audio prefetch + caching are handled inside comparison/evaluate.py
# (manifest-driven for the first, parquet-backed for the second) so re-runs
# skip already-downloaded shards.
#
# Datasets:
#   - umutkkgz/tr-full-dataset                   (~large, manifest-driven)
#   - Martingkc/processed_audio_tr-with-emotions (155 clips, parquet-backed)
#
# Usage:
#   experiments/comparison/run_turkish.sh <max_samples> [checkpoint...]
#
# Examples:
#   experiments/comparison/run_turkish.sh 500
#   experiments/comparison/run_turkish.sh 500 \
#       results/exp1-hubert-b1-b2_20260421-102709/finetune/best_model_finetuned.pt
#
# Env:
#   E2V_HUB=ms|hf   FunASR hub for emotion2vec variants (default ms)
set -euo pipefail

MAX_SAMPLES=${1:?"usage: $0 <max_samples> [checkpoint...]"}
shift
OUR_CHECKPOINTS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATASETS=(
  "umutkkgz/tr-full-dataset"
  "Martingkc/processed_audio_tr-with-emotions"
)

for ds in "${DATASETS[@]}"; do
  echo "================================================================"
  echo "=== dataset: $ds"
  echo "================================================================"
  experiments/comparison/run_hf_dataset.sh "$ds" "$MAX_SAMPLES" "${OUR_CHECKPOINTS[@]}"
done
