#!/usr/bin/env bash
# Run Qwen2-Audio (local HF load) against a single source.
#
# Usage:
#   experiments/comparison/run_qwen.sh --split-dir  <path> [max_samples]
#   experiments/comparison/run_qwen.sh --hf-dataset <id>   [max_samples]
#
# Env:
#   QWEN_MODEL   HF model id (default Qwen/Qwen2-Audio-7B-Instruct)
set -euo pipefail

SRC_FLAG=${1:?"usage: $0 --split-dir|--hf-dataset <arg> [max_samples]"}
SRC_ARG=${2:?"usage: $0 --split-dir|--hf-dataset <arg> [max_samples]"}
MAX_SAMPLES=${3:-}

case "$SRC_FLAG" in
  --split-dir|--hf-dataset) ;;
  *) echo "first arg must be --split-dir or --hf-dataset, got: $SRC_FLAG" >&2; exit 1;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2-Audio-7B-Instruct}"

EXTRA=()
if [[ -n "$MAX_SAMPLES" ]]; then
  EXTRA+=(--max-samples "$MAX_SAMPLES")
fi

echo "=== qwen: $QWEN_MODEL on $SRC_FLAG $SRC_ARG ==="
python comparison/evaluate.py \
  "$SRC_FLAG" "$SRC_ARG" \
  --qwen \
  --qwen-model "$QWEN_MODEL" \
  "${EXTRA[@]}"
