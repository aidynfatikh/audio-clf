from pathlib import Path
import os
import sys
import warnings

# Disable torchcodec and use soundfile for audio decoding
os.environ['DATASETS_AUDIO_BACKEND'] = 'soundfile'
os.environ['TORCHCODEC_QUIET'] = '1'
warnings.filterwarnings('ignore', category=UserWarning)

from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent
DOTENV = REPO_ROOT / ".env"
DATASET_ID = "01gumano1d/batch01-aug"
DATA_DIR = REPO_ROOT / "data" / "batch01-aug"

load_dotenv(DOTENV)
HF_TOKEN = os.environ.get("HF_TOKEN")


def read_audio(row_audio):
    """Read audio from row['audio'] dict. Handles bytes or path (resolves in HF cache if needed)."""
    import io
    if not isinstance(row_audio, dict):
        raise ValueError("row_audio must be dict with 'bytes' or 'path'")
    if "bytes" in row_audio and row_audio["bytes"] is not None:
        import soundfile as sf
        return sf.read(io.BytesIO(row_audio["bytes"]))
    path = row_audio.get("path")
    if not path:
        raise ValueError("row_audio has no 'bytes' or 'path'")
    if not os.path.isabs(path):
        name = Path(path).name
        for base in (DATA_DIR, Path.home() / ".cache" / "huggingface" / "datasets"):
            if base.exists():
                found = list(base.rglob(name))
                if found:
                    path = str(found[0])
                    break
    import soundfile as sf
    return sf.read(path)


def load():
    """Load dataset. Installs (downloads) if not already cached."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set. Add HF_TOKEN=... to .env or env.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading dataset...")
    login(token=HF_TOKEN)
    return load_dataset(DATASET_ID, cache_dir=str(DATA_DIR))


def get_row(split_name: str = "train", index: int = 0, dataset=None):
    """Return a single row from the dataset with audio kept as raw bytes/path (not decoded).

    Args:
        split_name: Dataset split to use (e.g. 'train', 'validation', 'test').
        index:      Row index within the split.
        dataset:    Optional pre-loaded DatasetDict. Loaded automatically if None.

    Returns:
        dict with all columns for the requested row.
    """
    from datasets import Audio
    if dataset is None:
        dataset = load()
    split = dataset[split_name]
    if "audio" in split.column_names:
        split = split.cast_column("audio", Audio(decode=False))
    return split[index]


if __name__ == "__main__":
    try:
        ds = load()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print("Loaded", DATASET_ID)
    if hasattr(ds, "keys"):
        for k in ds.keys():
            part = ds[k]
            print(f"  {k}: {len(part)} rows")
            if len(part) > 0:
                print(f"      columns: {part.column_names}")
    else:
        print(f"  {len(ds)} rows", getattr(ds, "column_names", ""))
