"""Multi-task SSL-audio model with swappable backbone (HuBERT, WavLM, ...)."""

import torch
import torch.nn as nn
from transformers import HubertModel, WavLMModel


_BACKBONES: dict[str, type] = {
    "hubert": HubertModel,
    "wavlm": WavLMModel,
}


def _family(backbone_name: str) -> str:
    """Normalize YAML names like 'hubert_base', 'wavlm_base_plus' to a registry key."""
    n = (backbone_name or "").lower()
    for key in _BACKBONES:
        if n.startswith(key):
            return key
    raise ValueError(f"Unknown backbone '{backbone_name}'. Known: {list(_BACKBONES)}")


def _weighted_pool(
    all_layers: torch.Tensor,
    layer_weights: torch.Tensor,
    time_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax-weighted sum over hidden layers, then masked mean-pool over time.

    Args:
        all_layers:    [num_layers, B, T, H]  – stacked hidden states
        layer_weights: [num_layers]           – learnable unnormalised weights
        time_mask:     [B, T] bool, True for valid frames. If None, plain mean.
    Returns:
        [B, H] pooled features
    """
    w = torch.softmax(layer_weights, dim=0)
    pooled = (w.view(-1, 1, 1, 1) * all_layers).sum(dim=0)  # [B, T, H]
    if time_mask is None:
        return pooled.mean(dim=1)
    m = time_mask.to(pooled.dtype).unsqueeze(-1)            # [B, T, 1]
    summed = (pooled * m).sum(dim=1)                        # [B, H]
    denom = m.sum(dim=1).clamp(min=1.0)                     # [B, 1]
    return summed / denom


class MultiTaskBackbone(nn.Module):
    """SSL-audio encoder with three heads using task-specific weighted layer sums.

    Backbone is chosen at construction via `backbone_name` / `pretrained`.
    `hidden_size` and layer count are read from the backbone config, so HuBERT
    and WavLM both work without hardcoded dimensions.
    """

    def __init__(
        self,
        num_emotions,
        num_genders,
        num_ages,
        freeze_backbone: bool = True,
        use_spec_augment: bool = False,
        backbone_name: str = "hubert",
        pretrained: str = "facebook/hubert-base-ls960",
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained = pretrained

        cls = _BACKBONES[_family(backbone_name)]
        self.backbone = cls.from_pretrained(pretrained, output_hidden_states=True)

        # Prefer the model's built-in SpecAugment (masks projected features pre-transformer)
        # over manually masking post-transformer hidden states.
        self.use_spec_augment = use_spec_augment
        if hasattr(self.backbone, "config"):
            self.backbone.config.apply_spec_augment = bool(use_spec_augment)
            self.backbone.config.mask_time_prob = 0.05
            self.backbone.config.mask_time_length = 10
            self.backbone.config.mask_feature_prob = 0.0
            self.backbone.config.mask_feature_length = 10
        if hasattr(self.backbone.config, 'training_drop_path'):
            self.backbone.config.training_drop_path = 0.1

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = int(self.backbone.config.hidden_size)
        num_layers = int(self.backbone.config.num_hidden_layers) + 1  # +1 for embedding/projection hidden state

        # Learnable per-task layer weights (before softmax)
        self.emotion_weights = nn.Parameter(torch.ones(num_layers))
        self.gender_weights = nn.Parameter(torch.ones(num_layers))
        self.age_weights = nn.Parameter(torch.ones(num_layers))

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

    def forward(self, input_values, input_lengths=None):
        # NOTE: deliberately not passing attention_mask to the backbone — base
        # HuBERT/WavLM (feat_extract_norm="group") were pretrained without one,
        # and passing one shifts the pretrained activation distribution. The
        # padding mask only matters at the pooling stage.
        outputs = self.backbone(input_values)

        # Frozen backbone: stack once under no_grad. Heads/layer_weights still get grads.
        # Unfrozen (any param trainable): keep the graph so the backbone receives gradients.
        backbone_frozen = not any(p.requires_grad for p in self.backbone.parameters())
        if backbone_frozen:
            with torch.no_grad():
                all_layers = torch.stack(outputs.hidden_states, dim=0)  # [L, B, T, H]
        else:
            all_layers = torch.stack(outputs.hidden_states, dim=0)

        time_mask = None
        if input_lengths is not None:
            T = all_layers.shape[2]
            feat_lengths = self.backbone._get_feat_extract_output_lengths(input_lengths)
            feat_lengths = feat_lengths.to(input_values.device).long().clamp(max=T)
            idx = torch.arange(T, device=input_values.device).unsqueeze(0)  # [1, T]
            time_mask = idx < feat_lengths.unsqueeze(1)                     # [B, T]

        emo_feats = _weighted_pool(all_layers, self.emotion_weights, time_mask)
        gen_feats = _weighted_pool(all_layers, self.gender_weights, time_mask)
        age_feats = _weighted_pool(all_layers, self.age_weights, time_mask)

        return (
            self.emotion_head(emo_feats),
            self.gender_head(gen_feats),
            self.age_head(age_feats),
        )


# Backwards-compatible alias — existing imports `from multihead.model import MultiTaskHubert`
# keep working. Defaulting to hubert-base-ls960 preserves the prior behavior exactly.
MultiTaskHubert = MultiTaskBackbone
