"""Backward-compatibility shim.

All logic has moved to the top-level ``utils/`` package:
  utils.misc          — constants, set_seed, unwrap, sigint_handler
  utils.checkpointing — MODEL_DIR, save/rotate checkpoints, cosine schedule
  utils.data          — AudioDataset, build_label_encoders, HF dataset helpers
  utils.data_loading  — build_mixed_train_val_splits, env vars
  utils.training      — train_epoch, validate, make_batch_end_handler
  utils.audio_augment — NoiseMixer, speed_perturb, mix_at_snr
  utils.finetune_utils — layer analysis, discriminative-LR optimizer

Eval scripts that import from here continue to work unchanged.
"""

from utils.checkpointing import (
    MODEL_DIR,
    make_cosine_schedule,
    rotate_step_checkpoints,
    save_step_checkpoint,
    save_wandb_file_artifact,
)
from utils.data import (
    VAL_FRACTION,
    AudioDataset,
    _count_emotion_distribution,
    _count_label_presence,
    _ensure_label_columns,
    _force_canonical_label_schema,
    _prepare_split_for_training,
    build_label_encoders,
    fallback_split_train_val,
)
from utils.data_loading import (
    HF_BATCH01_CACHE,
    HF_BATCH01_ID,
    HF_BATCH01_SPLIT,
    HF_BATCH02_CACHE,
    HF_BATCH02_ID,
    KAZEMO_MAX_SAMPLES,
    KAZEMO_VAL_FRACTION,
    SPLIT_MANIFEST_DIR,
    TRAIN_VAL_MANIFEST,
    USE_BATCH01_TRAIN,
    USE_BATCH02,
    USE_KAZEMO,
    _resolve_train_val_manifest_path,
    build_holdout_mixed_train_val_splits,
    build_mixed_train_val_splits,
    build_splits_from_manifest_dir,
)
from utils.misc import (
    BATCH_SIZE_ENV_VAR,
    RANDOM_SEED,
    REPO_ROOT,
    SAMPLE_RATE,
    _ALL_TASKS,
    _KAZEMO_TASKS,
    apply_cuda_perf_flags,
    resolve_batch_size,
    set_seed,
    sigint_handler,
    stop_requested,
    unwrap,
)
from utils.training import (
    _VAL_METRIC_TO_WANDB,
    _wandb_val_keys,
    filter_val_metrics,
    make_batch_end_handler,
    train_epoch,
    validate,
)

# Legacy aliases kept for any code that referenced the private names.
_save_wandb_file_artifact = save_wandb_file_artifact
_save_step_checkpoint = save_step_checkpoint
_rotate_step_checkpoints = rotate_step_checkpoints
_make_cosine_schedule = make_cosine_schedule
_sigint_handler = sigint_handler
_unwrap = unwrap
