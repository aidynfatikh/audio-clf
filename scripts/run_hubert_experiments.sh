#!/usr/bin/env bash
# Run HuBERT stage-1 (train) + stage-2 (finetune) on all 3 split configs, GPU 1.
# Each experiment lands in results/<label>_<ts>/ with finetune/ subdir.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && set -a && source .env && set +a

export CUDA_VISIBLE_DEVICES=1
# Force HuBERT — .env may set BACKBONE=wavlm for another session.
export BACKBONE=hubert
TS=$(date +%Y%m%d-%H%M%S)

# label | split-dir | config
EXPERIMENTS=(
  "hubert-01         |batch01_only_v1|configs/splits/batch01_only.yaml"
  "hubert-01-02      |b1_b2_v1       |configs/splits/batch01_plus_batch02.yaml"
  "hubert-01-02-kazemo|b1_b2_kazemo_v1|configs/splits/all_three.yaml"
)

for entry in "${EXPERIMENTS[@]}"; do
  IFS="|" read -r LABEL SPLIT CONFIG <<<"$entry"
  LABEL=$(echo "$LABEL" | xargs); SPLIT=$(echo "$SPLIT" | xargs); CONFIG=$(echo "$CONFIG" | xargs)
  OUT="results/${LABEL}_${TS}"

  [[ -f "splits/${SPLIT}/summary.json" ]] || python3 scripts/build_splits.py --config "$CONFIG"

  mkdir -p "$OUT"
  export SPLIT_MANIFEST_DIR="splits/${SPLIT}"
  export MODEL_DIR="$OUT"

  echo "=== [$LABEL] STAGE 1 (train) → $OUT ==="
  WANDB_RUN_NAME="${LABEL}-${TS}-s1" \
    python3 multihead/train.py 2>&1 | tee "$OUT/train.log"

  echo "=== [$LABEL] STAGE 2 (finetune) → $OUT/finetune ==="
  WANDB_RUN_NAME="${LABEL}-${TS}-s2" \
    python3 multihead/finetune.py 2>&1 | tee "$OUT/finetune.log"
done

echo "=== All 3 experiments done ==="
