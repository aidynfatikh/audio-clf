# Audio Classification with Multi-Task HuBERT

Fine-tune HuBERT with multiple heads for emotion, gender, and age classification on audio data.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your HuggingFace token in `.env`:
```
HF_TOKEN=your_token_here
```

3. (Optional) If you see FFmpeg/torchcodec warnings, you can ignore them - the script uses `soundfile` and `librosa` for audio processing, not torchcodec. To suppress the warnings, the script automatically sets `TORCHCODEC_QUIET=1`.

## Usage

### Training

Run the training script:
```bash
python train.py
```

The script will:
- Load the dataset from `01gumano1d/batch01-aug`
- Automatically detect emotion and gender classes
- Train a frozen HuBERT backbone with three task-specific heads
- Save the best model to `models/best_model.pt`
- Save label encoders to `models/label_encoders.json`

### Configuration

You can modify these parameters in `train.py`:
- `BATCH_SIZE`: Batch size (default: 8)
- `LEARNING_RATE`: Learning rate for backbone (default: 1e-4)
- `HEAD_LEARNING_RATE`: Learning rate for heads (default: 1e-3)
- `NUM_EPOCHS`: Number of training epochs (default: 10)
- `EMOTION_WEIGHT`, `GENDER_WEIGHT`, `AGE_WEIGHT`: Loss weights for multi-task learning

### Model Architecture

- **Backbone**: HuBERT-base (frozen)
- **Emotion Head**: Classification with Cross-Entropy Loss
- **Gender Head**: Classification with Cross-Entropy Loss  
- **Age Head**: Regression with MSE Loss

### Features

- Automatic audio resampling to 16kHz (required by HuBERT)
- Weighted random sampling for balanced training
- Multi-task loss with configurable weights
- Layer-wise learning rate (lower for backbone, higher for heads)
- Validation metrics tracking
