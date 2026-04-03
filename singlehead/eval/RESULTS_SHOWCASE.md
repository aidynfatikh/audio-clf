Single-head (emotion) — results showcase

Model: HuBERT single-head emotion. Val / Test: 504 samples each.

——— Finetuned best ———

test:
emotion acc: 70.4%

val:
emotion acc: 73.6%

test emotion per class:
disgusted: 98.6%
fearful: 81.9%
surprised: 72.2%
angry: 69.4%
happy: 65.3%
sad: 62.5%
neutral: 43.1%

val emotion per class:
disgusted: 98.6%
fearful: 84.7%
surprised: 70.8%
happy: 73.6%
angry: 68.1%
sad: 66.7%
neutral: 52.8%

Checkpoint: single_head/models/emotion/finetune/best_model_finetuned.pt

——— Finetuned latest ———

test:
emotion acc: 75.6%

val:
emotion acc: 75.4%

test emotion per class:
disgusted: 100.0%
fearful: 80.6%
neutral: 75.0%
sad: 75.0%
happy: 65.3%
surprised: 69.4%
angry: 63.9%

val emotion per class:
disgusted: 100.0%
fearful: 86.1%
sad: 72.2%
surprised: 70.8%
angry: 69.4%
happy: 66.7%
neutral: 62.5%

Checkpoint: single_head/models/emotion/finetune/latest_checkpoint_finetune.pt
