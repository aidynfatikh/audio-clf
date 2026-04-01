---
language:
- en
library_name: pytorch
pipeline_tag: audio-classification
tags:
- audio-classification
- hubert
- multi-task
- speech
---

# hubert-baseline

Multi-task HuBERT model for speech attribute classification with three prediction heads: **emotion**, **gender**, and **age category**.

- **Backbone:** [`facebook/hubert-base-ls960`](https://huggingface.co/facebook/hubert-base-ls960)
- **Training dataset:** `01gumano1d/batch01-validation-test`
- **Audio input:** resampled to 16 kHz, clipped/padded to 10 s

## Training

Two-phase fine-tuning:

1. **Pretraining** — backbone frozen, heads only (10 epochs)
2. **Fine-tuning** — top 4 backbone layers unfrozen, lower LR (20 epochs)

## Results

Evaluated on the **train** split (`n = 7 646`).

### Overall Accuracy

| Task | Accuracy |
|---|---:|
| Emotion | 0.7935 |
| Gender | 0.9423 |
| Age | 0.8820 |

### Per-Class Accuracy

**Emotion**

| Class | Accuracy |
|---|---:|
| angry | 0.8777 |
| disgusted | 0.7720 |
| fearful | 0.7445 |
| happy | 0.8751 |
| neutral | 0.8765 |
| sad | 0.6019 |
| surprised | 0.5750 |

**Gender**

| Class | Accuracy |
|---|---:|
| F | 0.9049 |
| M | 0.9757 |

**Age**

| Class | Accuracy |
|---|---:|
| adult | 0.9230 |
| child | 0.9620 |
| senior | 0.9054 |
| young | 0.6506 |