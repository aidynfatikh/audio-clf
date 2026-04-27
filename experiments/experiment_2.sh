#!/usr/bin/env bash
# Experiment 2: WavLM × 3 compositions, augmentation ON (stage1 + stage2).
set -e
cd "$(dirname "$0")/.."
[[ -f .env ]] && set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export BACKBONE=wavlm
TS=$(date +%Y%m%d-%H%M%S)

# Shared eval manifest: every experiment validates and tests on the SAME rows.
EVAL_SPLIT_CFG=configs/splits/combined_eval_v1.yaml
EVAL_NAME=$(python3 -c 'import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))["name"])' "$EVAL_SPLIT_CFG")
[[ -f splits/$EVAL_NAME/summary.json ]] || python3 scripts/build_splits.py --config "$EVAL_SPLIT_CFG"
export EVAL_MANIFEST_DIR="splits/$EVAL_NAME"

run() {
  local label=$1 split_cfg=$2 train=$3 ft=$4
  local name; name=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['name'])" "$split_cfg")
  [[ -f splits/$name/summary.json ]] || python3 scripts/build_splits.py --config "$split_cfg"
  export SPLIT_MANIFEST_DIR="splits/$name" MODEL_DIR="results/${label}_${TS}"
  mkdir -p "$MODEL_DIR"
  TRAIN_CONFIG_NAME=$train WANDB_RUN_NAME="${label}-${TS}-train" python3 multihead/train.py
  [[ $ft != "-" ]] && TRAIN_CONFIG_NAME=$ft WANDB_RUN_NAME="${label}-${TS}-ft" python3 multihead/finetune.py
}

run exp2-wavlm-b1        configs/splits/batch01_only_aug.yaml         stage1 stage2
run exp2-wavlm-b1-b2     configs/splits/batch01_plus_batch02_aug.yaml stage1 stage2
run exp2-wavlm-b1-b2-kz  configs/splits/all_three_aug.yaml            stage1 stage2
