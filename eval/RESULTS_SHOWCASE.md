Audio classification — results (showcase)

Model: HuBERT multi-task (emotion + gender + age), finetuned best checkpoint.
Val / Test: 504 samples each.

test:
emotion acc: 74.6%
gender acc: 94.4%
age acc: 79.2%

val:
emotion acc: 77.0%
gender acc: 94.6%
age acc: 80.0%

test emotion per class:
disgusted: 97.2%
fearful: 80.6%
sad: 76.4%
happy: 70.8%
angry: 69.4%
surprised: 68.1%
neutral: 59.7%

val emotion per class:
disgusted: 100.0%
fearful: 84.7%
sad: 83.3%
happy: 70.8%
angry: 72.2%
surprised: 70.8%
neutral: 56.9%

test gender:
F: 94.8%
M: 94.0%

val gender:
F: 95.6%
M: 93.7%

test age:
child: 92.9%
senior: 88.9%
young: 71.4%
adult: 63.5%

val age:
child: 89.7%
senior: 85.7%
young: 81.0%
adult: 63.5%

Checkpoint: models/finetune/best_model_finetuned.pt
