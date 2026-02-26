#!/usr/bin/env python3
"""Fine-tune HuBERT with multiple heads (emotion, gender, age) on audio data."""

import os
import signal
import sys
import warnings

# Disable torchcodec and use soundfile for audio decoding
os.environ['DATASETS_AUDIO_BACKEND'] = 'soundfile'
os.environ['TORCHCODEC_QUIET'] = '1'
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from datasets import Audio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from collections import Counter

# Import load_data (warnings already suppressed in load_data.py)
from load_data import load, read_audio, DATA_DIR

# Configuration
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
HEAD_LEARNING_RATE = 1e-3  # Higher LR for new heads
NUM_EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000  # HuBERT requires 16kHz

# Loss weights for multi-task learning
EMOTION_WEIGHT = 1.0
GENDER_WEIGHT = 1.0
AGE_WEIGHT = 1.0

# Model save directory
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Safe stop: set by SIGINT handler (Ctrl+C)
_stop_requested = False


def _sigint_handler(signum, frame):
    global _stop_requested
    if _stop_requested:
        print("\nSecond Ctrl+C: exiting immediately.", file=sys.stderr)
        sys.exit(130)
    _stop_requested = True
    print("\nCtrl+C received. Finishing current batch and saving checkpoint...")


class MultiTaskHubert(nn.Module):
    """HuBERT model with three heads: emotion, gender, and age."""
    
    def __init__(self, num_emotions, num_genders, freeze_backbone=True):
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        
        # Freeze the backbone
        if freeze_backbone:
            for param in self.hubert.parameters():
                param.requires_grad = False
        
        hidden_size = 768  # Standard for Hubert-Base
        
        # Three separate heads
        self.emotion_head = nn.Linear(hidden_size, num_emotions)
        self.gender_head = nn.Linear(hidden_size, num_genders)
        self.age_head = nn.Linear(hidden_size, 1)  # Regression for age
    
    def forward(self, input_values):
        """
        Args:
            input_values: Audio features [batch_size, sequence_length]
        Returns:
            emotion_logits, gender_logits, age_prediction
        """
        outputs = self.hubert(input_values)
        
        # Mean pooling: [batch, time, 768] -> [batch, 768]
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        
        emotion_logits = self.emotion_head(embeddings)
        gender_logits = self.gender_head(embeddings)
        age_prediction = self.age_head(embeddings)
        
        return emotion_logits, gender_logits, age_prediction


class AudioDataset(Dataset):
    """Dataset for audio classification with emotion, gender, and age labels."""
    
    def __init__(self, dataset_split, processor, emotion_encoder, gender_encoder, 
                 max_length=160000):  # 10 seconds at 16kHz
        self.data = dataset_split
        self.processor = processor
        self.emotion_encoder = emotion_encoder
        self.gender_encoder = gender_encoder
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        
        # Read audio
        audio_data, sample_rate = read_audio(row['audio'])
        
        # Resample to 16kHz if needed
        if sample_rate != SAMPLE_RATE:
            import librosa
            audio_data = librosa.resample(
                audio_data.astype(np.float32),
                orig_sr=sample_rate,
                target_sr=SAMPLE_RATE
            )
        
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Truncate or pad to max_length
        if len(audio_data) > self.max_length:
            audio_data = audio_data[:self.max_length]
        else:
            audio_data = np.pad(audio_data, (0, self.max_length - len(audio_data)))
        
        # Process with HuBERT processor
        inputs = self.processor(
            audio_data,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values.squeeze(0)
        
        # Get labels
        emotion_label = self.emotion_encoder.get(row.get('emotion', 'unknown'), -1)
        gender_label = self.gender_encoder.get(row.get('gender', 'unknown'), -1)
        age_label = float(row.get('age', 0.0))
        
        return {
            'input_values': input_values,
            'emotion': torch.tensor(emotion_label, dtype=torch.long),
            'gender': torch.tensor(gender_label, dtype=torch.long),
            'age': torch.tensor(age_label, dtype=torch.float32)
        }


def build_label_encoders(dataset):
    """Build label encoders for emotion and gender from the dataset."""
    emotions = set()
    genders = set()
    
    # Collect all unique labels (disable audio decoding to avoid torchcodec)
    for split_name in dataset.keys():
        split = dataset[split_name]
        # Remove audio decoding by accessing only label columns
        split_no_audio = split.remove_columns(['audio']) if 'audio' in split.column_names else split
        for row in split_no_audio:
            if 'emotion' in row and row['emotion'] is not None:
                emotions.add(str(row['emotion']))
            if 'gender' in row and row['gender'] is not None:
                genders.add(str(row['gender']))
    
    # Create encoders (label -> index)
    emotion_encoder = {label: idx for idx, label in enumerate(sorted(emotions))}
    gender_encoder = {label: idx for idx, label in enumerate(sorted(genders))}
    
    # Add unknown for missing labels
    emotion_encoder['unknown'] = len(emotion_encoder)
    gender_encoder['unknown'] = len(gender_encoder)
    
    return emotion_encoder, gender_encoder


def compute_class_weights(dataset, label_key, encoder):
    """Compute class weights for balanced sampling."""
    labels = []
    for split_name in dataset.keys():
        split = dataset[split_name]
        # Remove audio decoding to avoid torchcodec
        split_no_audio = split.remove_columns(['audio']) if 'audio' in split.column_names else split
        for row in split_no_audio:
            label = str(row.get(label_key, 'unknown'))
            labels.append(encoder.get(label, encoder['unknown']))
    
    counter = Counter(labels)
    total = len(labels)
    num_classes = len(encoder)
    
    weights = []
    for i in range(num_classes):
        count = counter.get(i, 1)  # Avoid division by zero
        weights.append(total / (num_classes * count))
    
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, dataloader, criterion_emotion, criterion_gender, 
                criterion_age, optimizer, device):
    """Train for one epoch. Returns (metrics_dict, was_stopped)."""
    global _stop_requested
    model.train()
    total_loss = 0.0
    emotion_loss_sum = 0.0
    gender_loss_sum = 0.0
    age_loss_sum = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        if _stop_requested:
            break
        input_values = batch['input_values'].to(device)
        emotion_labels = batch['emotion'].to(device)
        gender_labels = batch['gender'].to(device)
        age_labels = batch['age'].to(device)
        
        optimizer.zero_grad()
        
        emotion_logits, gender_logits, age_pred = model(input_values)
        
        # Compute losses
        loss_emotion = criterion_emotion(emotion_logits, emotion_labels)
        loss_gender = criterion_gender(gender_logits, gender_labels)
        loss_age = criterion_age(age_pred.squeeze(), age_labels)
        
        # Combined loss
        total_batch_loss = (
            EMOTION_WEIGHT * loss_emotion +
            GENDER_WEIGHT * loss_gender +
            AGE_WEIGHT * loss_age
        )
        
        total_batch_loss.backward()
        optimizer.step()
        
        total_loss += total_batch_loss.item()
        emotion_loss_sum += loss_emotion.item()
        gender_loss_sum += loss_gender.item()
        age_loss_sum += loss_age.item()
        num_batches += 1

    if num_batches == 0:
        num_batches = 1
    return {
        'total': total_loss / num_batches,
        'emotion': emotion_loss_sum / num_batches,
        'gender': gender_loss_sum / num_batches,
        'age': age_loss_sum / num_batches
    }, _stop_requested


def validate(model, dataloader, criterion_emotion, criterion_gender, 
             criterion_age, device):
    """Validate the model."""
    global _stop_requested
    model.eval()
    total_loss = 0.0
    emotion_loss_sum = 0.0
    gender_loss_sum = 0.0
    age_loss_sum = 0.0
    
    emotion_correct = 0
    gender_correct = 0
    age_mae = 0.0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if _stop_requested:
                break
            input_values = batch['input_values'].to(device)
            emotion_labels = batch['emotion'].to(device)
            gender_labels = batch['gender'].to(device)
            age_labels = batch['age'].to(device)
            
            emotion_logits, gender_logits, age_pred = model(input_values)
            
            # Compute losses
            loss_emotion = criterion_emotion(emotion_logits, emotion_labels)
            loss_gender = criterion_gender(gender_logits, gender_labels)
            loss_age = criterion_age(age_pred.squeeze(), age_labels)
            
            total_batch_loss = (
                EMOTION_WEIGHT * loss_emotion +
                GENDER_WEIGHT * loss_gender +
                AGE_WEIGHT * loss_age
            )
            
            total_loss += total_batch_loss.item()
            emotion_loss_sum += loss_emotion.item()
            gender_loss_sum += loss_gender.item()
            age_loss_sum += loss_age.item()
            
            # Compute accuracy
            emotion_pred = torch.argmax(emotion_logits, dim=1)
            gender_pred = torch.argmax(gender_logits, dim=1)
            
            emotion_correct += (emotion_pred == emotion_labels).sum().item()
            gender_correct += (gender_pred == gender_labels).sum().item()
            age_mae += torch.abs(age_pred.squeeze() - age_labels).sum().item()
            total_samples += emotion_labels.size(0)
            num_batches += 1

    if num_batches == 0:
        num_batches = 1
    if total_samples == 0:
        total_samples = 1
    return {
        'total': total_loss / num_batches,
        'emotion': emotion_loss_sum / num_batches,
        'gender': gender_loss_sum / num_batches,
        'age': age_loss_sum / num_batches,
        'emotion_acc': emotion_correct / total_samples,
        'gender_acc': gender_correct / total_samples,
        'age_mae': age_mae / total_samples
    }


def main():
    global _stop_requested
    signal.signal(signal.SIGINT, _sigint_handler)
    print(f"Using device: {DEVICE}")
    print("Press Ctrl+C to stop training and save a checkpoint.")

    # Load dataset
    dataset = load()
    
    # Build label encoders
    print("Building label encoders...")
    emotion_encoder, gender_encoder = build_label_encoders(dataset)
    num_emotions = len(emotion_encoder)
    num_genders = len(gender_encoder)
    
    print(f"Found {num_emotions} emotion classes: {list(emotion_encoder.keys())}")
    print(f"Found {num_genders} gender classes: {list(gender_encoder.keys())}")
    
    # Save encoders
    encoder_path = MODEL_DIR / "label_encoders.json"
    with open(encoder_path, 'w') as f:
        json.dump({
            'emotion': emotion_encoder,
            'gender': gender_encoder
        }, f, indent=2)
    print(f"Saved label encoders to {encoder_path}")
    
    # Initialize feature extractor
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    
    # Create datasets (disable automatic audio decoding)
    train_split = dataset.get('train', dataset.get('train', None))
    if train_split is None:
        # If no train split, use the first available split
        train_split = dataset[list(dataset.keys())[0]]
    
    # Disable automatic audio decoding (we handle it manually with read_audio)
    if 'audio' in train_split.column_names:
        train_split = train_split.cast_column('audio', Audio(decode=False))
    
    val_split = dataset.get('validation', dataset.get('val', dataset.get('test', None)))
    if val_split is not None and 'audio' in val_split.column_names:
        val_split = val_split.cast_column('audio', Audio(decode=False))
    
    train_dataset = AudioDataset(
        train_split,
        processor,
        emotion_encoder,
        gender_encoder
    )
    
    if val_split is not None:
        val_dataset = AudioDataset(
            val_split,
            processor,
            emotion_encoder,
            gender_encoder
        )
    else:
        val_dataset = None
        print("Warning: No validation split found. Using train split for validation.")
        val_dataset = train_dataset
    
    # Compute class weights for balanced sampling
    emotion_weights = compute_class_weights(dataset, 'emotion', emotion_encoder)
    gender_weights = compute_class_weights(dataset, 'gender', gender_encoder)
    
    # Create weighted sampler (using emotion weights as primary)
    sample_weights = [emotion_weights[train_dataset[i]['emotion'].item()] 
                     for i in range(len(train_dataset))]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    # Initialize model
    print("Initializing model...")
    model = MultiTaskHubert(
        num_emotions=num_emotions,
        num_genders=num_genders,
        freeze_backbone=True
    ).to(DEVICE)
    
    # Setup optimizer with different learning rates
    # Backbone (frozen, but we'll set LR for potential unfreezing later)
    backbone_params = list(model.hubert.parameters())
    head_params = list(model.emotion_head.parameters()) + \
                  list(model.gender_head.parameters()) + \
                  list(model.age_head.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},  # Lower LR for backbone
        {'params': head_params, 'lr': HEAD_LEARNING_RATE}
    ], weight_decay=0.01)
    
    # Loss functions
    criterion_emotion = nn.CrossEntropyLoss(weight=emotion_weights.to(DEVICE))
    criterion_gender = nn.CrossEntropyLoss(weight=gender_weights.to(DEVICE))
    criterion_age = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    last_epoch = -1

    for epoch in range(NUM_EPOCHS):
        if _stop_requested:
            break
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)

        # Train
        train_metrics, stopped = train_epoch(
            model, train_loader, criterion_emotion, criterion_gender,
            criterion_age, optimizer, DEVICE
        )
        last_epoch = epoch

        print(f"Train Loss - Total: {train_metrics['total']:.4f}, "
              f"Emotion: {train_metrics['emotion']:.4f}, "
              f"Gender: {train_metrics['gender']:.4f}, "
              f"Age: {train_metrics['age']:.4f}")
        if stopped:
            break

        # Validate
        val_metrics = validate(
            model, val_loader, criterion_emotion, criterion_gender,
            criterion_age, DEVICE
        )
        if _stop_requested:
            break

        print(f"Val Loss - Total: {val_metrics['total']:.4f}, "
              f"Emotion: {val_metrics['emotion']:.4f}, "
              f"Gender: {val_metrics['gender']:.4f}, "
              f"Age: {val_metrics['age']:.4f}")
        print(f"Val Metrics - Emotion Acc: {val_metrics['emotion_acc']:.4f}, "
              f"Gender Acc: {val_metrics['gender_acc']:.4f}, "
              f"Age MAE: {val_metrics['age_mae']:.2f}")

        # Save best model
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            model_path = MODEL_DIR / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'num_emotions': num_emotions,
                'num_genders': num_genders,
            }, model_path)
            print(f"Saved best model to {model_path}")

    if _stop_requested:
        checkpoint_path = MODEL_DIR / "checkpoint_interrupted.pt"
        torch.save({
            'epoch': last_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'num_emotions': num_emotions,
            'num_genders': num_genders,
        }, checkpoint_path)
        print(f"\nStopped by user. Checkpoint saved to {checkpoint_path}")
    else:
        print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
