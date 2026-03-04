# Audio Classification with Multi-Task HuBERT

Fine-tune HuBERT with multiple heads for emotion, gender, and age classification on audio data.

## Setup

1. Install dependencies:
```bash
source /home/fatikh/ML/ML/bin/activate   # or your venv
pip install -r requirements.txt
```

2. Set up your HuggingFace token in `.env`:
```
HF_TOKEN=your_token_here
```

3. (Optional) FFmpeg/torchcodec warnings can be ignored; the pipeline uses `soundfile` and `librosa` for audio. `TORCHCODEC_QUIET=1` is set automatically.

## Usage

### Training

**Stage 1 — frozen backbone**
```bash
python train.py
```
- Loads dataset from `01gumano1d/batch01-aug`, infers emotion/gender/age classes.
- Trains with a frozen HuBERT backbone and three task heads.
- Saves best model to `models/best_model.pt` and label encoders to `models/label_encoders.json`.

**Stage 2 — fine-tune top layers** (optional)
```bash
python finetune.py
```
- Loads stage-1 checkpoint, unfreezes the top-ranked transformer layers (see below), trains with discriminative LRs.
- Saves to `models/finetune/`. Use `python finetune.py --analyze` to print layer importance and exit.

### Generate test audio (ElevenLabs)

Generate 16 kHz WAVs from a text file for testing the classifier. Uses ElevenLabs multilingual TTS and picks voices by gender/age.

1. Get an API key from [ElevenLabs](https://elevenlabs.io/app/settings/api-keys) and set it:
   ```bash
   export ELEVENLABS_API_KEY=your_key
   ```
   Or add `ELEVENLABS_API_KEY=...` to `.env`.

2. Create a txt file with one line per utterance:
   ```
   M,child,neutral: Үстелде тыныш қана сурет салып отырмын [calm].
   F,child,neutral: Сабақтан кейін дәптерлерімді реттеп қойдым [steady tone].
   M,young,neutral: Аялдамада автобусты күтіп тұрмын [neutral].
   ```

3. Run (with your venv activated):
   ```bash
   source /home/fatikh/ML/ML/bin/activate
   python scripts/generate_test_audio_elevenlabs.py scripts/sample_input.txt -o generated_audio
   ```
   Output is written to `generated_audio/` as 16 kHz WAVs. Use `--model eleven_multilingual_v2` (default) or `eleven_turbo_v2_5`; see `--help` for options.

### Configuration

Edit constants in code as needed:
- **train.py**: `BATCH_SIZE`, `HEAD_LEARNING_RATE`, `NUM_EPOCHS`, `EMOTION_WEIGHT`, `GENDER_WEIGHT`, `AGE_WEIGHT`.
- **finetune.py**: `UNFREEZE_TOP_N`, `UNFREEZE_FEATURE_PROJ`, `BACKBONE_LR_TOP`, `LAYER_DECAY`, `HEAD_LR`, `NUM_EPOCHS`, `BATCH_SIZE`.

## Model Architecture

- **Backbone**: HuBERT-base (`facebook/hubert-base-ls960`), 16 kHz input.
- **Representation**: All 13 hidden states (feature-projection output + 12 transformer layers) combined via task-specific learned weights, then mean-pooled over time (see Training techniques).
- **Heads**: Emotion, gender, and age are each a 768→256→num_classes MLP (ReLU; dropout on emotion). All tasks use cross-entropy loss.

## Training Techniques

**Layer weighting (stage 1)**  
Each task has its own learnable weights over the 13 HuBERT hidden states. A softmax over these weights gives a convex combination of layers; the result is mean-pooled over time and fed to the task head. This lets the model emphasize different layers per task (e.g. emotion vs gender) without fixing a single “best” layer.

**Layer selection for fine-tuning (stage 2)**  
Transformer layers are ranked by importance using the **final** stage-1 layer weights (from the last epoch of the latest `training_metrics*.json`). Importance is averaged across the three tasks. Only the **top-N** layers in this ranking are unfrozen (default N=4); the rest stay frozen. Optionally, the feature-projection layer can be unfrozen as well.

**Discriminative learning rates (stage 2)**  
Unfrozen encoder layers use a per-layer learning rate: **LR_n = backbone_lr_top × ξ^(12−n)**, where n is the layer index (0–11) and ξ is the decay factor (default 0.85). Higher layers (closer to the output) get a larger LR; lower layers a smaller one to avoid overwriting general representations. The three heads and the learnable layer weights use a single, higher LR (`head_lr`). If the feature-projection is unfrozen, it uses the same formula with the strongest decay (as if it were layer −1).

**Other details**  
- Stage 1: only the heads and the 13 per-task layer weights are trained; the backbone is frozen. AdamW with weight decay; optional `torch.compile` for throughput.
- Multi-task loss: weighted sum of cross-entropy losses (emotion, gender, age) with fixed task weights.
- Audio is resampled to 16 kHz and clipped/padded to 10 s per sample.

## Features

- Automatic 16 kHz resampling and 10 s fixed-length segments.
- Multi-task loss with configurable task weights.
- Checkpoint resume (stage 1 and stage 2); Ctrl+C saves and allows clean stop.
- Validation metrics and best-model saving.
