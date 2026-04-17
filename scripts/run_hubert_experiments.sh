#!/usr/bin/env bash
# Run HuBERT multi-head training on all 3 split configs sequentially on GPU 1.
# Each run's outputs land in results/hubert/<label>_<ts>/.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && set -a && source .env && set +a

export CUDA_VISIBLE_DEVICES=1
TS=$(date +%Y%m%d-%H%M%S)

# label | split-dir | config
EXPERIMENTS=(
  "batch01|batch01_only_v1|configs/splits/batch01_only.yaml"
  "b1b2  |b1_b2_v1       |configs/splits/batch01_plus_batch02.yaml"
  "b1b2kz|b1_b2_kazemo_v1|configs/splits/all_three.yaml"
)

for entry in "${EXPERIMENTS[@]}"; do
  IFS="|" read -r LABEL SPLIT CONFIG <<<"$entry"
  LABEL=$(echo "$LABEL" | xargs); SPLIT=$(echo "$SPLIT" | xargs); CONFIG=$(echo "$CONFIG" | xargs)
  OUT="results/hubert/${LABEL}_${TS}"

  [[ -f "splits/${SPLIT}/summary.json" ]] || python3 scripts/build_splits.py --config "$CONFIG"

  echo "=== [$LABEL] split=$SPLIT → $OUT ==="
  mkdir -p "$OUT"
  SPLIT_MANIFEST_DIR="splits/${SPLIT}" \
  MODEL_DIR="$OUT" \
  WANDB_RUN_NAME="hubert-${LABEL}-${TS}" \
    python3 multihead/train.py 2>&1 | tee "$OUT/train.log"
done

echo "=== All 3 experiments done ==="
