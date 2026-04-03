"""Shared paths for evaluation artifacts (under repo ``results/``)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
