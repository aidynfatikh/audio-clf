"""Extract the `augmented` flag from a raw HF row.

The flag may appear at the top level (`augmented` / `is_augmented`) or nested
under a `metadata` dict or JSON string. Returns None when the column is absent
so the caller can distinguish "not augmented" from "don't know".
"""

from __future__ import annotations

import json
import re
from typing import Any

_TOP_LEVEL_KEYS = ("augmented", "is_augmented")
_NESTED_KEYS = ("augmented", "is_augmented")


def _to_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _parse_metadata(md: Any) -> dict[str, Any]:
    if isinstance(md, dict):
        return md
    if isinstance(md, str):
        text = md.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            out: dict[str, Any] = {}
            m = re.search(r'"?augmented"?\s*[:=]\s*(true|false)', text, flags=re.I)
            if m:
                out["augmented"] = m.group(1).lower() == "true"
            return out
    return {}


def extract_augmented_flag(row: dict[str, Any]) -> bool | None:
    """Return True/False if known, None if no `augmented` signal is present."""
    for key in _TOP_LEVEL_KEYS:
        if key in row:
            b = _to_bool(row.get(key))
            if b is not None:
                return b

    md = _parse_metadata(row.get("metadata"))
    for key in _NESTED_KEYS:
        if key in md:
            b = _to_bool(md.get(key))
            if b is not None:
                return b

    return None


def is_augmented(row: dict[str, Any]) -> bool:
    """True only when the flag is known AND equals True."""
    return extract_augmented_flag(row) is True
