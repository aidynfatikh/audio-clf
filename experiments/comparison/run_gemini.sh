#!/usr/bin/env bash
# Run Gemini API against a single source (split-dir OR HF dataset).
#
# Usage:
#   experiments/comparison/run_gemini.sh --split-dir  <path>    [max_samples]
#   experiments/comparison/run_gemini.sh --hf-dataset <id>      [max_samples]
#
# Env:
#   GEMINI_MODEL       model id passed via --gemini-model (default gemini-3.1-pro-preview)
#   GEMINI_CONCURRENCY parallel in-flight API calls (default 4)
#   GEMINI_API_KEY     required; read from .env by evaluate.py if unset here
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

GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.1-pro-preview}"
GEMINI_CONCURRENCY="${GEMINI_CONCURRENCY:-4}"

EXTRA=()
if [[ -n "$MAX_SAMPLES" ]]; then
  EXTRA+=(--max-samples "$MAX_SAMPLES")
fi

echo "=== gemini: $GEMINI_MODEL (concurrency=$GEMINI_CONCURRENCY) on $SRC_FLAG $SRC_ARG ==="
python comparison/evaluate.py \
  "$SRC_FLAG" "$SRC_ARG" \
  --gemini \
  --gemini-model "$GEMINI_MODEL" \
  --gemini-concurrency "$GEMINI_CONCURRENCY" \
  "${EXTRA[@]}"
