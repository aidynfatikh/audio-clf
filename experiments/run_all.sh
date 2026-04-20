#!/usr/bin/env bash
# Run experiments 1-4 sequentially. Fail-fast: a crash in any experiment
# aborts the rest so the outputs stay consistent.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

for i in 1 2 3 4; do
  echo "=== [$(date '+%F %T')] starting experiment_${i}.sh ==="
  bash "$HERE/experiment_${i}.sh"
  echo "=== [$(date '+%F %T')] finished experiment_${i}.sh ==="
done

echo "All experiments complete."
