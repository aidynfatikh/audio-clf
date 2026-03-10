#!/usr/bin/env python3
"""Generate test WAVs with ElevenLabs from a txt file.

Input format: GENDER,age,neutral: Sentence [emotion].
With model eleven_v3, [brackets] are AUDIO TAGS (expression only, not spoken).
We move a trailing [tag] to the start so the whole sentence gets that expression,
e.g. "Hello world [calm]." -> "[calm] Hello world."  Use --model eleven_v3 for expressions.
Output: WAVs in el_audios/; JSON/CSV in el_audios/info/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import wave
from pathlib import Path
from types import SimpleNamespace

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from elevenlabs import ElevenLabs
from elevenlabs.types import VoiceSettings

OUTPUT_FORMAT = "pcm_16000"
# Lower stability = more creative/expressive (0.35); higher = more monotone
STABILITY_CREATIVE = 0.0
SAMPLE_RATE = 16000
# (gender, age) -> (voice_id, name). Only young, adult, senior (no child voices).
# All IDs and names are taken from voices.json.
VOICE_BY_GENDER_AGE = {
    # Female
    ("F", "young"): ("21m00Tcm4TlvDq8ikWAM", "Rachel"),
    ("F", "adult"): ("9BWtsMINqrJLrRacOk9x", "Aria"),
    ("F", "senior"): ("RILOU7YmBhvwJGDGjNmP", "Jane"),  # age=old
    # Male
    ("M", "young"): ("CYw3kZ02Hs0563khs1Fj", "Dave"),
    ("M", "adult"): ("29vD33N1CtxCmqQRPOHJ", "Drew"),
    ("M", "senior"): ("ZQe5CZNOzWyzPSCn5a3c", "James"),  # age=old
}

# (gender, emotion) -> (voice_id, name), for stronger expressions when available.
# Do not use library voices: Kannada, Ivanna, Lucy, Pete, Jeff, Poe.
VOICE_BY_GENDER_EMOTION = {
    ("F", "neutral"): ("LcfcDJNUP1GQjkzn1xUU", "Emily"),
    ("M", "neutral"): ("onwK4e9ZLuTAKqWW03F9", "Daniel - Steady Broadcaster"),
    ("F", "happy"): ("cgSgspJ2msm6clMCkdW9", "Jessica - Playful, Bright, Warm"),
    ("M", "happy"): ("bIHbv24MWmeRgasZH58o", "Will - Relaxed Optimist"),
    ("F", "sad"): ("MF3mGyEYCl7XYWbV9V6O", "Elli"),
    ("M", "sad"): ("flq6f7yk4E4fJM5XTYuZ", "Michael"),
    ("F", "angry"): ("AZnzlk1XvdvUeBnXmlld", "Domi"),
    ("M", "angry"): ("ODq5zmih8GrVes37Dizd", "Patrick"),  # shouty — not library Poe
    ("F", "fearful"): ("piTKgcLEGmPE4e6mEKli", "Nicole"),
    ("M", "fearful"): ("GBv7mTt0atIp3Br8iCZE", "Thomas"),
    ("F", "surprised"): ("jsCqWAovK2LkecY7zXl4", "Freya"),
    ("M", "surprised"): ("bVMeCyTHy58xNoL34h3p", "Jeremy"),
    ("F", "disgusted"): ("FGY2WhTYpPnrIDTdsKH5", "Laura - Enthusiast, Quirky Attitude"),
    ("M", "disgusted"): ("t0jbNlBVZ17f02VDIeMI", "Jessie"),
}


def parse_line(line: str) -> tuple[str, str, str, str] | None:
    """Parse 'M,child,neutral: text' -> (gender, age, emotion, text)."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    m = re.match(r"^([MF]),\s*(\w+),\s*(\w+):\s*(.+)$", line, re.IGNORECASE)
    if not m:
        return None
    gender = m.group(1).upper()
    age = m.group(2).strip().lower()
    emotion = m.group(3).strip().lower()
    text = m.group(4).strip()
    return (gender, age, emotion, text) if text else None


def text_for_tts(text: str, model_id: str) -> str:
    """For eleven_v3: move trailing [tag] to the start so the model applies expression to the whole sentence."""
    if "eleven_v3" not in model_id:
        return text
    # Match trailing [optional space and period after] [...] at end
    m = re.search(r"\s*\[([^\]]+)\]\s*\.?\s*$", text)
    if not m:
        return text
    tag = m.group(1).strip()
    rest = text[: m.start()].strip().rstrip(".")
    if not rest:
        return text
    return f"[{tag}] {rest}."


def get_client(api_key: str | None = None) -> ElevenLabs:
    key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise ValueError("Set ELEVENLABS_API_KEY or pass --api-key.")
    return ElevenLabs(api_key=key)


def list_voices(client: ElevenLabs | None = None, voices_path: Path | None = None):
    """Load voices from a local JSON file (voices.json), not from the ElevenLabs API.

    The file is expected to be a list of dicts with keys:
    voice_id, name, category, labels.
    """
    if voices_path is None:
        # Repo root has voices.json; this script lives in scripts/
        voices_path = Path(__file__).resolve().parent.parent / "voices.json"
    if not voices_path.exists():
        raise SystemExit(f"voices.json not found at {voices_path}")
    raw = json.loads(voices_path.read_text(encoding="utf-8"))
    # Wrap dicts so the rest of the code can use attribute access.
    return [SimpleNamespace(**v) for v in raw]


def voice_to_dict(v) -> dict:
    labels = getattr(v, "labels", None) or {}
    if hasattr(labels, "model_dump"):
        labels = labels.model_dump()
    elif not isinstance(labels, dict):
        labels = {k: getattr(labels, k, None) for k in ("gender", "age", "accent", "description", "use_case") if getattr(labels, k, None)}
    return {"voice_id": getattr(v, "voice_id", None), "name": getattr(v, "name", None), "category": getattr(v, "category", None), "labels": labels}


def pick_voice_for(voices, gender: str, age: str, emotion: str = "") -> tuple[str, str] | None:
    # Prefer emotion-matched voice when defined (better expression)
    if emotion:
        em_key = (gender, emotion.strip().lower())
        if em_key in VOICE_BY_GENDER_EMOTION:
            return VOICE_BY_GENDER_EMOTION[em_key]
    # No child voices: map child -> young for age-based lookup
    age_key = "young" if age == "child" else age
    key = (gender, age_key)
    if key in VOICE_BY_GENDER_AGE:
        return VOICE_BY_GENDER_AGE[key]
    g = "female" if gender == "F" else "male"
    name_fn = lambda v: getattr(v, "name", None) or (getattr(v, "voice_id", "") or "")[:8]
    for v in voices:
        lb = getattr(v, "labels", None)
        if lb is None:
            continue
        vg = (lb.get("gender") if isinstance(lb, dict) else getattr(lb, "gender", None)) or ""
        if g in str(vg).lower():
            return (v.voice_id, name_fn(v))
    return (voices[0].voice_id, name_fn(voices[0])) if voices else None


def pcm_to_wav(pcm: bytes, path: str, rate: int = 16000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def generate_one(client: ElevenLabs, voice_id: str, text: str, out_path: str, model_id: str, output_format: str):
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
        voice_settings=VoiceSettings(stability=STABILITY_CREATIVE),
    )
    if isinstance(audio, bytes):
        pcm = audio
    elif hasattr(audio, "read"):
        pcm = audio.read()
    else:
        pcm = b"".join(audio)
    pcm_to_wav(pcm, out_path, rate=SAMPLE_RATE)


def save_voices_info(voices, info_dir: Path):
    all_data = [voice_to_dict(v) for v in voices]
    (info_dir / "all_voices.json").write_text(json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Generate test WAVs with ElevenLabs (voice by gender/age).")
    p.add_argument("input_txt", type=Path, help="Txt: lines like GENDER,age,neutral: Sentence [emotion].")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("el_audios"), help="Output dir; info in <dir>/info")
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--model", type=str, default="eleven_v3", help="eleven_v3 for [bracket] expression tags; eleven_multilingual_v2 for no tags")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.input_txt.exists():
        raise SystemExit(f"Not found: {args.input_txt}")

    rows = []
    for i, line in enumerate(args.input_txt.read_text(encoding="utf-8").splitlines()):
        parsed = parse_line(line)
        if parsed:
            rows.append((i + 1, parsed[0], parsed[1], parsed[2], parsed[3]))  # idx, gender, age, emotion, text
    if not rows:
        raise SystemExit("No valid lines.")

    print(f"Parsed {len(rows)} line(s).")
    if args.dry_run:
        for idx, gender, age, emotion, text in rows:
            print(f"  {idx}: {gender},{age},{emotion} -> {text[:60]}{'...' if len(text) > 60 else ''}")
        return

    client = get_client(args.api_key)
    voices = list_voices(client)
    out = args.output_dir
    info_dir = out / "info"
    out.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    save_voices_info(voices, info_dir)
    print(f"Model: {args.model}, voices: {len(voices)}. Info -> {info_dir}")

    log_rows = []
    test_pairs = []
    for idx, gender, age, emotion, text in rows:
        voice_id, voice_name = pick_voice_for(voices, gender, age, emotion) or (None, None)
        if not voice_id:
            raise SystemExit("No voice available.")
        out_name = f"line_{idx:03d}_{gender}_{age}.wav"
        out_path = out / out_name
        tts_text = text_for_tts(text, args.model)
        print(f"  {out_name} -> \"{voice_name}\" ... ", end="", flush=True)
        generate_one(client, voice_id, tts_text, str(out_path), model_id=args.model, output_format=OUTPUT_FORMAT)
        print("ok")
        log_rows.append({"line": idx, "output_file": out_name, "gender": gender, "age": age, "emotion": emotion, "voice_id": voice_id, "voice_name": voice_name})
        # Testing pairs: (path, gender, age, emotion) for evaluation
        rel_path = out_name  # or str(out_path) for absolute
        test_pairs.append({"path": rel_path, "gender": gender, "age": age, "emotion": emotion})

    (info_dir / "voices_used.json").write_text(json.dumps(log_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (info_dir / "test_pairs.json").write_text(json.dumps(test_pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. WAVs -> {out.absolute()}, info -> {info_dir.absolute()}")
    print(f"Info: all_voices.json, voices_used.json, test_pairs.json ({len(test_pairs)} pairs)")


if __name__ == "__main__":
    main()
