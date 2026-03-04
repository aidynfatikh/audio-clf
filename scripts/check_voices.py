#!/usr/bin/env python3
"""Fetch ElevenLabs voices and write them to a JSON file."""
from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from elevenlabs import ElevenLabs


def voice_to_dict(v) -> dict:
    labels = getattr(v, "labels", None) or {}
    if hasattr(labels, "model_dump"):
        labels = labels.model_dump()
    elif not isinstance(labels, dict):
        labels = {
            k: getattr(labels, k, None)
            for k in ("gender", "age", "accent", "description", "use_case")
            if getattr(labels, k, None)
        }
    return {
        "voice_id": getattr(v, "voice_id", None),
        "name": getattr(v, "name", None),
        "category": getattr(v, "category", None),
        "labels": labels,
    }


def main():
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise SystemExit("Set ELEVENLABS_API_KEY (e.g. in .env).")
    client = ElevenLabs(api_key=key)
    r = client.voices.get_all(show_legacy=True)
    voices = r.voices if hasattr(r, "voices") else r
    data = [voice_to_dict(v) for v in voices]
    out_path = Path(__file__).resolve().parent.parent / "voices.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {len(data)} voices to {out_path}")


if __name__ == "__main__":
    main()
