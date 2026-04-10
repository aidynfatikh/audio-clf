1. Experiment on batch01 + batch02 mixed dataset. Val: non augmented 1008 balanced set from batch01, not included in train data.

Final training metrics after 10 epochs pre-train + 20 epochs finetune.

"train": {
    "total": 0.485824,
    "emotion": 0.5924,
    "gender": 0.270866,
    "age": 0.465411,
    "emotion_acc": 0.94072,
    "gender_acc": 0.959163,
    "age_acc": 0.944122
},
"val": {
    "total": 1.311113,
    "emotion": 1.790384,
    "gender": 0.574021,
    "age": 1.104533,
    "emotion_acc": 0.574405,
    "gender_acc": 0.828373,
    "age_acc": 0.702381
},

Evaluation on KazEmoTTS balanced set.
"n": 10000,
"accuracy": 0.2176,
"top2_accuracy": 0.3966,
"per_class_accuracy": {
    "angry": 0.16076784643071385,
    "fearful": 0.016796640671865627,
    "happy": 0.1265746850629874,
    "neutral": 0.6862627474505099,
    "sad": 0.3067226890756303,
    "surprised": 0.008403361344537815
}

2. Experiment on batch01 only dataset. It was trained on the complete batch01 dataset and validated on it, so val is irrelevant.

Final training metrics after 10 epochs pre-train + 20 epochs finetune.

"train": {
    "total": 1.69876,
    "emotion": 0.837447,
    "gender": 0.292563,
    "age": 0.547542
},
"val": {
    "total": 1.747164,
    "emotion": 0.847767,
    "gender": 0.302277,
    "age": 0.578705,
    "emotion_acc": 0.918127,
    "gender_acc": 0.969396,
    "age_acc": 0.944154
},

Evaluation on KazEmoTTS balanced set.
"n": 10000,
"accuracy": 0.1837,
"top2_accuracy": 0.4316,
"per_class_accuracy": {
    "angry": 0.04259148170365927,
    "fearful": 0.008998200359928014,
    "happy": 0.026994601079784044,
    "neutral": 0.9844031193761248,
    "sad": 0.03361344537815126,
    "surprised": 0.005402160864345739
},

3. Experiment on batch02 only dataset with a 0.1 val split.

Final training metrics after 10 epochs pre-train + 10 epochs finetune.

"train": {
    "total": 0.68032,
    "emotion": 0.866089,
    "gender": 0.321339,
    "age": 0.636887
    },
"val": {
    "total": 0.728235,
    "emotion": 0.937412,
    "gender": 0.341023,
    "age": 0.670828,
    "emotion_acc": 0.785749,
    "gender_acc": 0.916253,
    "age_acc": 0.83715
},

Evaluation on KazEmoTTS balanced set.
"n": 10000,
"accuracy": 0.2207,
"top2_accuracy": 0.4265,
"per_class_accuracy": {
    "angry": 0.10377924415116976,
    "fearful": 0.008398320335932814,
    "happy": 0.13377324535092983,
    "neutral": 0.7270545890821836,
    "sad": 0.3469387755102041,
    "surprised": 0.004201680672268907
}