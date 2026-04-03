"""Multi-task HuBERT model for emotion, gender, and age classification."""

import torch
import torch.nn as nn
from transformers import HubertModel


def _weighted_pool(all_layers: torch.Tensor, layer_weights: torch.Tensor) -> torch.Tensor:
    """Softmax-weighted sum over HuBERT hidden layers, then mean-pool over time.

    Args:
        all_layers:    [num_layers, B, T, H]  – stacked hidden states
        layer_weights: [num_layers]           – learnable unnormalised weights
    Returns:
        [B, H] pooled features
    """
    w = torch.softmax(layer_weights, dim=0)
    pooled = (w.view(-1, 1, 1, 1) * all_layers).sum(dim=0)  # [B, T, H]
    return pooled.mean(dim=1)                                # [B, H]


class MultiTaskHubert(nn.Module):
    """HuBERT model with three heads using task-specific weighted layer sums."""

    def __init__(self, num_emotions, num_genders, num_ages, freeze_backbone=True, use_spec_augment=False):
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=True,)
        # Prefer the model's built-in SpecAugment (masks projected features pre-transformer)
        # over manually masking post-transformer hidden states.
        self.use_spec_augment = use_spec_augment
        if hasattr(self.hubert, "config"):
            self.hubert.config.apply_spec_augment = bool(use_spec_augment)
            # Reasonable defaults (HF wav2vec2/HubERT-style)
            self.hubert.config.mask_time_prob = 0.05
            self.hubert.config.mask_time_length = 10
            self.hubert.config.mask_feature_prob = 0.0
            self.hubert.config.mask_feature_length = 10
        if hasattr(self.hubert.config, 'training_drop_path'):
            self.hubert.config.training_drop_path = 0.1

        if freeze_backbone:
            for param in self.hubert.parameters():
                param.requires_grad = False

        hidden_size = 768
        num_layers = 13  # 12 transformer layers + initial embedding layer

        # Learnable per-task layer weights (before softmax)
        self.emotion_weights = nn.Parameter(torch.ones(num_layers))
        self.gender_weights = nn.Parameter(torch.ones(num_layers))
        self.age_weights = nn.Parameter(torch.ones(num_layers))

        # Slightly deeper heads with non-linearity
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_emotions),
        )
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_genders),
        )
        self.age_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_ages),
        )

    def forward(self, input_values, attention_mask=None):
        """
        Args:
            input_values: Audio features [batch_size, sequence_length]
        Returns:
            emotion_logits, gender_logits, age_logits
        """
        outputs = self.hubert(input_values, attention_mask=attention_mask)

        # hidden_states: tuple of 13 tensors [batch, time, hidden].
        # Frozen backbone: stack once under no_grad. Per-layer .detach().clone() before
        # stack duplicated every layer (~2× activation memory for this block); finetune
        # only stacks, which is why stage-1 looked heavier despite smaller batches.
        # Heads/layer_weights still get grads: they only multiply the stacked tensor.
        # Unfrozen: stack keeps the graph so backbone receives gradients.
        # False as soon as *any* HuBERT param is trainable (e.g. top-N finetune), not "all or nothing".
        backbone_frozen = not any(p.requires_grad for p in self.hubert.parameters())
        if backbone_frozen:
            with torch.no_grad():
                all_layers = torch.stack(outputs.hidden_states, dim=0)  # [13, B, T, H]
        else:
            all_layers = torch.stack(outputs.hidden_states, dim=0)

        emo_feats = _weighted_pool(all_layers, self.emotion_weights)
        gen_feats = _weighted_pool(all_layers, self.gender_weights)
        age_feats = _weighted_pool(all_layers, self.age_weights)

        return (
            self.emotion_head(emo_feats),
            self.gender_head(gen_feats),
            self.age_head(age_feats),
        )
