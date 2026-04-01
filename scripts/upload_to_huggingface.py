#!/usr/bin/env python3
"""Upload trained checkpoints and metadata to Hugging Face Hub.

This script exports the existing PyTorch checkpoint package used by this repo:
- checkpoint (.pt)
- label encoders JSON
- optional training metrics JSON
- generated export_config.json
- generated model card (README.md)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, create_repo


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
FINETUNE_DIR = MODELS_DIR / "finetune"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload model artifacts to Hugging Face Hub")
    parser.add_argument("--repo-name", required=True, help="Target model repo name")
    parser.add_argument("--org-name", default="", help="HF organization name (optional)")
    parser.add_argument(
        "--checkpoint",
        default="best",
        choices=["best", "latest", "interrupted", "best-finetuned", "latest-finetuned", "path"],
        help="Checkpoint source to upload",
    )
    parser.add_argument("--checkpoint-path", default="", help="Explicit checkpoint path when --checkpoint=path")
    parser.add_argument("--private", action="store_true", help="Create/private repo")
    parser.add_argument("--revision", default="main", help="Target revision/branch")
    parser.add_argument("--commit-message", default="Upload model export package", help="Commit message")
    parser.add_argument("--include-metrics", action="store_true", help="Include metrics JSON in upload package")
    parser.add_argument(
        "--model-card-path",
        default="",
        help="Optional custom model card template path. If set, this content becomes README.md",
    )
    parser.add_argument(
        "--metrics-path",
        default="",
        help="Optional explicit metrics JSON path (overrides auto-detection)",
    )
    parser.add_argument(
        "--label-encoders-path",
        default="models/label_encoders.json",
        help="Path to label_encoders.json for the selected model",
    )
    parser.add_argument(
        "--tags",
        default="audio-classification,multi-task,hubert",
        help="Comma-separated model tags for model card",
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""), help="HF token (or set HF_TOKEN)")
    return parser.parse_args()


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint == "path":
        if not args.checkpoint_path:
            raise ValueError("--checkpoint-path is required when --checkpoint=path")
        path = Path(args.checkpoint_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path

    mapping = {
        "best": MODELS_DIR / "best_model.pt",
        "latest": MODELS_DIR / "latest_checkpoint.pt",
        "interrupted": MODELS_DIR / "checkpoint_interrupted.pt",
        "best-finetuned": FINETUNE_DIR / "best_model_finetuned.pt",
        "latest-finetuned": FINETUNE_DIR / "latest_checkpoint_finetune.pt",
    }
    return mapping[args.checkpoint]


def read_label_encoders() -> dict:
    path = MODELS_DIR / "label_encoders.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def read_label_encoders_from_path(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def resolve_optional_path(path_str: str) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def resolve_metrics_path(checkpoint_path: Path, explicit_metrics_path: str = "") -> Path | None:
    explicit = resolve_optional_path(explicit_metrics_path)
    if explicit is not None:
        return explicit if explicit.exists() else None
    if "finetune" in str(checkpoint_path):
        p = FINETUNE_DIR / "training_metrics_finetune.json"
    else:
        p = MODELS_DIR / "training_metrics.json"
    return p if p.exists() else None


def build_export_config(checkpoint_path: Path, repo_id: str, label_encoders: dict) -> dict:
    return {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": repo_id,
        "source_checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)) if checkpoint_path.is_relative_to(REPO_ROOT) else str(checkpoint_path),
        "task": "audio-classification",
        "architecture": "MultiTaskHubert",
        "backbone": "facebook/hubert-base-ls960",
        "input": {
            "sample_rate": 16000,
            "max_length_samples": 160000,
            "channels": "mono",
        },
        "outputs": {
            "emotion_labels": sorted((label_encoders.get("emotion") or {}).keys()),
            "gender_labels": sorted((label_encoders.get("gender") or {}).keys()),
            "age_labels": sorted((label_encoders.get("age") or {}).keys()),
        },
    }


def build_model_card(repo_id: str, tags: list[str], checkpoint_name: str, include_metrics: bool) -> str:
    yaml_tags = "\n".join(f"- {t}" for t in tags if t)
    metrics_line = "Includes training metrics JSON for reproducibility." if include_metrics else "Training metrics JSON not included in this export."
    return f"""---
language:
- en
library_name: pytorch
tags:
{yaml_tags}
pipeline_tag: audio-classification
---

# {repo_id}

PyTorch checkpoint package exported from the `audio-clf` project.

## Contents
- `{checkpoint_name}`: model checkpoint payload (`state_dict`, optimizer state, metadata)
- `label_encoders.json`: task label encoders for emotion/gender/age
- `export_config.json`: input/output and export metadata
- `README.md`: this model card

{metrics_line}

## Notes
- This package keeps the repository's native checkpoint format.
- Use project inference/training code to load checkpoint and encoders.
"""


def build_model_card_from_template(
    template_path: Path,
    *,
    repo_id: str,
    checkpoint_name: str,
    include_metrics: bool,
) -> str:
    with open(template_path) as f:
        content = f.read()

    metrics_line = (
        "Includes training metrics JSON for reproducibility."
        if include_metrics
        else "Training metrics JSON not included in this export."
    )
    return (
        content
        .replace("{{REPO_ID}}", repo_id)
        .replace("{{CHECKPOINT_NAME}}", checkpoint_name)
        .replace("{{INCLUDE_METRICS_LINE}}", metrics_line)
    )


def main() -> None:
    args = parse_args()
    if not args.hf_token:
        raise RuntimeError("HF token missing. Set HF_TOKEN or pass --hf-token.")

    checkpoint_path = resolve_checkpoint_path(args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    label_encoders_path = resolve_optional_path(args.label_encoders_path)
    if label_encoders_path is None:
        raise RuntimeError("label encoders path resolution failed")

    owner = args.org_name.strip()
    repo_id = f"{owner}/{args.repo_name}" if owner else args.repo_name

    api = HfApi(token=args.hf_token)
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        token=args.hf_token,
        private=args.private,
        exist_ok=True,
    )

    label_encoders = read_label_encoders_from_path(label_encoders_path)
    metrics_path = resolve_metrics_path(checkpoint_path, args.metrics_path)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    with tempfile.TemporaryDirectory(prefix="hf-export-") as tmp:
        export_dir = Path(tmp)

        ckpt_target = export_dir / checkpoint_path.name
        shutil.copy2(checkpoint_path, ckpt_target)

        if label_encoders_path.exists():
            shutil.copy2(label_encoders_path, export_dir / "label_encoders.json")

        if args.include_metrics and metrics_path is not None:
            shutil.copy2(metrics_path, export_dir / metrics_path.name)

        export_config = build_export_config(checkpoint_path, repo_id, label_encoders)
        with open(export_dir / "export_config.json", "w") as f:
            json.dump(export_config, f, indent=2)

        include_metrics = bool(args.include_metrics and metrics_path is not None)
        template_path = resolve_optional_path(args.model_card_path)
        if template_path is not None:
            if not template_path.exists():
                raise FileNotFoundError(f"Model card template not found: {template_path}")
            model_card = build_model_card_from_template(
                template_path,
                repo_id=repo_id,
                checkpoint_name=checkpoint_path.name,
                include_metrics=include_metrics,
            )
        else:
            model_card = build_model_card(
                repo_id=repo_id,
                tags=tags,
                checkpoint_name=checkpoint_path.name,
                include_metrics=include_metrics,
            )
        with open(export_dir / "README.md", "w") as f:
            f.write(model_card)

        api.upload_folder(
            folder_path=str(export_dir),
            repo_id=repo_id,
            repo_type="model",
            token=args.hf_token,
            revision=args.revision,
            commit_message=args.commit_message,
        )

    print(f"Uploaded export package to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
