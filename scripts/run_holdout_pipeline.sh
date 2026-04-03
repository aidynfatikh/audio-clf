#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
set -a && source .env && set +a

t=$(date +%Y%m%d-%H%M%S)
export WANDB_RUN_NAME="stage1-$t"
python3 multihead/train.py

sleep 30

export WANDB_RUN_NAME="stage2-$t"
python3 multihead/finetune.py
