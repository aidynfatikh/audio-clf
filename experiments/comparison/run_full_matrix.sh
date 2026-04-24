#!/usr/bin/env bash
# Run the full comparison matrix: all models × (our split-dir + Turkish HF datasets).
# A failure in any single (model, source) pair is logged but doesn't abort the run —
# all other results still land on disk for the aggregator.
#
# Usage:
#   experiments/comparison/run_full_matrix.sh <split_dir> <max_samples> [checkpoint...]
#
# Example:
#   experiments/comparison/run_full_matrix.sh splits/b1_b2_noaug_v1 500 \
#     results/exp1-hubert-b1-b2_20260421-102709/finetune/best_model_finetuned.pt
#
# Env passthrough:
#   E2V_HUB, GEMINI_MODEL, GEMINI_CONCURRENCY, GEMINI_API_KEY, QWEN_MODEL
set -euo pipefail

SPLIT_DIR=${1:?"usage: $0 <split_dir> <max_samples> [checkpoint...]"}
MAX_SAMPLES=${2:?"usage: $0 <split_dir> <max_samples> [checkpoint...]"}
shift 2
OUR_CHECKPOINTS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

HF_DATASETS=(
  "umutkkgz/tr-full-dataset"
  "Martingkc/processed_audio_tr-with-emotions"
)

# (source_flag, source_arg) pairs.
SOURCES=(
  "--split-dir|$SPLIT_DIR"
)
for ds in "${HF_DATASETS[@]}"; do
  SOURCES+=("--hf-dataset|$ds")
done

run_or_warn() {
  local label="$1"; shift
  echo "────────────────────────────────────────────────────────────────"
  echo ">>> $label"
  echo "────────────────────────────────────────────────────────────────"
  if ! "$@"; then
    echo "!!! $label FAILED — continuing with the rest of the matrix." >&2
  fi
}

for pair in "${SOURCES[@]}"; do
  SRC_FLAG="${pair%%|*}"
  SRC_ARG="${pair##*|}"

  # emotion2vec variants (4 of them) — reuse the existing per-dataset runner.
  # run_hf_dataset.sh / run_emotion2vec.sh already enumerate variants.
  if [[ "$SRC_FLAG" == "--split-dir" ]]; then
    run_or_warn "emotion2vec × $SRC_ARG" \
      experiments/comparison/run_emotion2vec.sh "$SRC_ARG" "$MAX_SAMPLES"
  else
    run_or_warn "emotion2vec × $SRC_ARG" \
      experiments/comparison/run_hf_dataset.sh "$SRC_ARG" "$MAX_SAMPLES"
  fi

  # Gemini
  run_or_warn "gemini × $SRC_ARG" \
    experiments/comparison/run_gemini.sh "$SRC_FLAG" "$SRC_ARG" "$MAX_SAMPLES"

  # Qwen2-Audio
  run_or_warn "qwen × $SRC_ARG" \
    experiments/comparison/run_qwen.sh "$SRC_FLAG" "$SRC_ARG" "$MAX_SAMPLES"

  # Our checkpoint(s)
  for ckpt in "${OUR_CHECKPOINTS[@]}"; do
    run_or_warn "checkpoint=$ckpt × $SRC_ARG" \
      python comparison/evaluate.py "$SRC_FLAG" "$SRC_ARG" \
        --checkpoint "$ckpt" --max-samples "$MAX_SAMPLES"
  done
done

echo "================================================================"
echo ">>> Aggregating results"
echo "================================================================"
python scripts/comparison_summary.py
