"""Multi-task train/validate loops and the batch-end callback factory."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from utils.checkpointing import (
    rotate_step_checkpoints,
    save_step_checkpoint,
    save_wandb_file_artifact,
)
from utils.misc import _ALL_TASKS, stop_requested, unwrap

try:
    import wandb
except Exception:
    wandb = None

_KAZEMO_TASKS: frozenset = frozenset({"emotion"})

_VAL_METRIC_TO_WANDB = {
    "total": "loss_total",
    "emotion": "loss_emotion",
    "gender": "loss_gender",
    "age": "loss_age",
    "emotion_acc": "acc_emotion",
    "gender_acc": "acc_gender",
    "age_acc": "acc_age",
}


def _wandb_val_keys(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}/{_VAL_METRIC_TO_WANDB[k]}": float(v) for k, v in metrics.items()}


def filter_val_metrics(metrics: dict, tasks) -> dict:
    """Return a copy of *metrics* with only the keys relevant to *tasks*."""
    out: dict = {
        "total": metrics["total"],
        "emotion": metrics["emotion"],
        "emotion_acc": metrics["emotion_acc"],
    }
    if "gender" in tasks:
        out["gender"] = metrics["gender"]
        out["gender_acc"] = metrics["gender_acc"]
    if "age" in tasks:
        out["age"] = metrics["age"]
        out["age_acc"] = metrics["age_acc"]
    return out


def train_epoch(
    model,
    dataloader,
    criterion_emotion,
    criterion_gender,
    criterion_age,
    optimizer,
    device,
    *,
    step_state=None,
    on_batch_end=None,
    epoch_index: int = -1,
    emotion_weight: float = 1.2,
    gender_weight: float = 0.5,
    age_weight: float = 1.0,
    grad_clip_norm: float = 1.0,
):
    """Train for one epoch. Returns (metrics_dict, was_stopped)."""
    global stop_requested
    model.train()
    total_loss = emotion_loss_sum = gender_loss_sum = age_loss_sum = 0.0
    emotion_loss_batches = gender_loss_batches = age_loss_batches = num_batches = 0
    emotion_correct = gender_correct = age_correct = 0
    emotion_samples = gender_samples = age_samples = 0

    _use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if _use_bf16 else torch.float32

    for batch in tqdm(dataloader, desc="Training"):
        if stop_requested:
            break
        input_values = batch["input_values"].to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
        emotion_labels = batch["emotion"].to(device, non_blocking=True)
        gender_labels = batch["gender"].to(device, non_blocking=True)
        age_labels = batch["age"].to(device, non_blocking=True)
        has_emotion = batch["has_emotion"].to(device, non_blocking=True)
        has_gender = batch["has_gender"].to(device, non_blocking=True)
        has_age = batch["has_age"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            emotion_logits, gender_logits, age_logits = model(
                input_values, attention_mask=attention_mask
            )
            per_task_losses: dict = {}
            active_weight = 0.0
            total_batch_loss = torch.zeros((), device=device)

            if has_emotion.any():
                loss = criterion_emotion(emotion_logits[has_emotion], emotion_labels[has_emotion])
                per_task_losses["emotion"] = loss
                total_batch_loss = total_batch_loss + emotion_weight * loss
                active_weight += emotion_weight
            if has_gender.any():
                loss = criterion_gender(gender_logits[has_gender], gender_labels[has_gender])
                per_task_losses["gender"] = loss
                total_batch_loss = total_batch_loss + gender_weight * loss
                active_weight += gender_weight
            if has_age.any():
                loss = criterion_age(age_logits[has_age], age_labels[has_age])
                per_task_losses["age"] = loss
                total_batch_loss = total_batch_loss + age_weight * loss
                active_weight += age_weight

            if active_weight > 0.0:
                total_batch_loss = total_batch_loss / active_weight
            else:
                continue

            if has_emotion.any():
                emotion_correct += (torch.argmax(emotion_logits[has_emotion], 1) == emotion_labels[has_emotion]).sum().item()
                emotion_samples += int(has_emotion.sum())
            if has_gender.any():
                gender_correct += (torch.argmax(gender_logits[has_gender], 1) == gender_labels[has_gender]).sum().item()
                gender_samples += int(has_gender.sum())
            if has_age.any():
                age_correct += (torch.argmax(age_logits[has_age], 1) == age_labels[has_age]).sum().item()
                age_samples += int(has_age.sum())

        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        current_step = None
        if step_state is not None:
            step_state["global_step"] = int(step_state.get("global_step", 0)) + 1
            step_state["samples_seen"] = int(step_state.get("samples_seen", 0)) + int(input_values.shape[0])
            current_step = int(step_state["global_step"])

        if on_batch_end is not None and current_step is not None:
            on_batch_end({
                "global_step": current_step,
                "epoch": epoch_index,
                "train_total_loss": float(total_batch_loss.item()),
                "train_emotion_loss": float(per_task_losses["emotion"].item()) if "emotion" in per_task_losses else None,
                "train_gender_loss": float(per_task_losses["gender"].item()) if "gender" in per_task_losses else None,
                "train_age_loss": float(per_task_losses["age"].item()) if "age" in per_task_losses else None,
            })

        total_loss += total_batch_loss.item()
        emotion_loss_sum += float(per_task_losses["emotion"].item()) if "emotion" in per_task_losses else 0.0
        gender_loss_sum += float(per_task_losses["gender"].item()) if "gender" in per_task_losses else 0.0
        age_loss_sum += float(per_task_losses["age"].item()) if "age" in per_task_losses else 0.0
        emotion_loss_batches += "emotion" in per_task_losses
        gender_loss_batches += "gender" in per_task_losses
        age_loss_batches += "age" in per_task_losses
        num_batches += 1

    num_batches = max(num_batches, 1)
    return {
        "total": total_loss / num_batches,
        "emotion": emotion_loss_sum / max(emotion_loss_batches, 1),
        "gender": gender_loss_sum / max(gender_loss_batches, 1),
        "age": age_loss_sum / max(age_loss_batches, 1),
        "emotion_acc": emotion_correct / max(emotion_samples, 1),
        "gender_acc": gender_correct / max(gender_samples, 1),
        "age_acc": age_correct / max(age_samples, 1),
    }, stop_requested


def validate(
    model,
    dataloader,
    criterion_emotion,
    criterion_gender,
    criterion_age,
    device,
    *,
    emotion_weight: float = 1.2,
    gender_weight: float = 0.5,
    age_weight: float = 1.0,
):
    global stop_requested
    model.eval()
    total_loss = emotion_loss_sum = gender_loss_sum = age_loss_sum = 0.0
    emotion_loss_batches = gender_loss_batches = age_loss_batches = num_batches = 0
    emotion_correct = gender_correct = age_correct = 0
    emotion_samples = gender_samples = age_samples = 0

    _use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if _use_bf16 else torch.float32

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if stop_requested:
                break
            input_values = batch["input_values"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            emotion_labels = batch["emotion"].to(device, non_blocking=True)
            gender_labels = batch["gender"].to(device, non_blocking=True)
            age_labels = batch["age"].to(device, non_blocking=True)
            has_emotion = batch["has_emotion"].to(device, non_blocking=True)
            has_gender = batch["has_gender"].to(device, non_blocking=True)
            has_age = batch["has_age"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                emotion_logits, gender_logits, age_logits = model(
                    input_values, attention_mask=attention_mask
                )
                per_task_losses: dict = {}
                active_weight = 0.0
                total_batch_loss = torch.zeros((), device=device)

                if has_emotion.any():
                    loss = criterion_emotion(emotion_logits[has_emotion], emotion_labels[has_emotion])
                    per_task_losses["emotion"] = loss
                    total_batch_loss = total_batch_loss + emotion_weight * loss
                    active_weight += emotion_weight
                if has_gender.any():
                    loss = criterion_gender(gender_logits[has_gender], gender_labels[has_gender])
                    per_task_losses["gender"] = loss
                    total_batch_loss = total_batch_loss + gender_weight * loss
                    active_weight += gender_weight
                if has_age.any():
                    loss = criterion_age(age_logits[has_age], age_labels[has_age])
                    per_task_losses["age"] = loss
                    total_batch_loss = total_batch_loss + age_weight * loss
                    active_weight += age_weight

                if active_weight > 0.0:
                    total_batch_loss = total_batch_loss / active_weight
                else:
                    continue

            total_loss += total_batch_loss.item()
            emotion_loss_sum += float(per_task_losses["emotion"].item()) if "emotion" in per_task_losses else 0.0
            gender_loss_sum += float(per_task_losses["gender"].item()) if "gender" in per_task_losses else 0.0
            age_loss_sum += float(per_task_losses["age"].item()) if "age" in per_task_losses else 0.0
            emotion_loss_batches += "emotion" in per_task_losses
            gender_loss_batches += "gender" in per_task_losses
            age_loss_batches += "age" in per_task_losses

            if has_emotion.any():
                emotion_correct += (torch.argmax(emotion_logits[has_emotion], 1) == emotion_labels[has_emotion]).sum().item()
                emotion_samples += int(has_emotion.sum())
            if has_gender.any():
                gender_correct += (torch.argmax(gender_logits[has_gender], 1) == gender_labels[has_gender]).sum().item()
                gender_samples += int(has_gender.sum())
            if has_age.any():
                age_correct += (torch.argmax(age_logits[has_age], 1) == age_labels[has_age]).sum().item()
                age_samples += int(has_age.sum())
            num_batches += 1

    num_batches = max(num_batches, 1)
    return {
        "total": total_loss / num_batches,
        "emotion": emotion_loss_sum / max(emotion_loss_batches, 1),
        "gender": gender_loss_sum / max(gender_loss_batches, 1),
        "age": age_loss_sum / max(age_loss_batches, 1),
        "emotion_acc": emotion_correct / max(emotion_samples, 1),
        "gender_acc": gender_correct / max(gender_samples, 1),
        "age_acc": age_correct / max(age_samples, 1),
    }


def make_batch_end_handler(
    *,
    step_state: dict,
    train_state: dict,
    all_step_val_metrics: list,
    model,
    optimizer,
    scheduler,
    num_emotions: int,
    num_genders: int,
    num_ages: int,
    checkpoint_every_steps: int,
    checkpoint_keep_last_n: int,
    checkpoint_save_latest_every_steps: bool,
    step_ckpt_dir: Path,
    latest_path: Path,
    step_val_metrics_path: Path,
    val_every_steps: int,
    val_loaders: dict,
    val_tasks: dict,
    criterion_emotion,
    criterion_gender,
    criterion_age,
    device,
    emotion_weight: float,
    gender_weight: float,
    age_weight: float,
    wandb_run,
    wandb_upload_step_artifact: bool,
    wandb_upload_latest_artifact: bool,
    wandb_latest_artifact_every_steps: int,
    step_artifact_name: str,
    latest_artifact_name: str,
):
    """Return an on_batch_end callback. All config captured at factory-call time."""

    def _handler(payload):
        global_step = payload["global_step"]

        if wandb_run is not None:
            data = {
                "train/loss_total": payload["train_total_loss"],
                "train/epoch": payload["epoch"] + 1,
            }
            if payload["train_emotion_loss"] is not None:
                data["train/loss_emotion"] = payload["train_emotion_loss"]
            if payload["train_gender_loss"] is not None:
                data["train/loss_gender"] = payload["train_gender_loss"]
            if payload["train_age_loss"] is not None:
                data["train/loss_age"] = payload["train_age_loss"]
            wandb_run.log(data, step=global_step)

        if checkpoint_every_steps > 0 and global_step % checkpoint_every_steps == 0:
            step_path = save_step_checkpoint(
                step_dir=step_ckpt_dir,
                global_step=global_step,
                epoch=payload["epoch"],
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=train_state["best_val_loss"],
                num_emotions=num_emotions,
                num_genders=num_genders,
                num_ages=num_ages,
                samples_seen=step_state["samples_seen"],
            )
            rotate_step_checkpoints(step_ckpt_dir, checkpoint_keep_last_n)
            print(f"Saved step checkpoint: {step_path.name}")

            if checkpoint_save_latest_every_steps:
                torch.save(
                    {
                        "epoch": payload["epoch"],
                        "global_step": global_step,
                        "samples_seen": step_state["samples_seen"],
                        "model_state_dict": unwrap(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": train_state["best_val_loss"],
                        "num_emotions": num_emotions,
                        "num_genders": num_genders,
                        "num_ages": num_ages,
                    },
                    latest_path,
                )

            if wandb_run is not None and wandb_upload_step_artifact:
                save_wandb_file_artifact(
                    wandb_run, file_path=step_path, name=step_artifact_name, artifact_type="checkpoint"
                )
            if (
                wandb_run is not None
                and wandb_upload_latest_artifact
                and wandb_latest_artifact_every_steps > 0
                and global_step % wandb_latest_artifact_every_steps == 0
                and latest_path.exists()
            ):
                save_wandb_file_artifact(
                    wandb_run, file_path=latest_path, name=latest_artifact_name, artifact_type="checkpoint"
                )

        if val_every_steps > 0 and global_step % val_every_steps == 0:
            step_record: dict = {"global_step": global_step, "epoch": payload["epoch"] + 1}
            wandb_data: dict = {}
            for _name, _vloader in val_loaders.items():
                _prefix = "val" if _name == "val" else f"val_{_name}"
                _tasks = val_tasks.get(_name, _ALL_TASKS)
                _sv = filter_val_metrics(
                    validate(
                        model, _vloader, criterion_emotion, criterion_gender,
                        criterion_age, device,
                        emotion_weight=emotion_weight,
                        gender_weight=gender_weight,
                        age_weight=age_weight,
                    ),
                    _tasks,
                )
                model.train()
                _print = (
                    f"  [Step {global_step}][{_name}] Val Loss: {_sv['total']:.4f}  "
                    f"Emotion Acc: {_sv['emotion_acc']:.4f}"
                )
                if "gender" in _tasks:
                    _print += f"  Gender Acc: {_sv['gender_acc']:.4f}"
                if "age" in _tasks:
                    _print += f"  Age Acc: {_sv['age_acc']:.4f}"
                print(_print)
                wandb_data.update(_wandb_val_keys(_prefix, _sv))
                step_record[_prefix] = {k: round(float(v), 6) for k, v in _sv.items()}
            if wandb_run is not None:
                wandb_run.log(wandb_data, step=global_step)
            all_step_val_metrics.append(step_record)
            with open(step_val_metrics_path, "w") as f:
                json.dump(all_step_val_metrics, f, indent=2)

    return _handler
