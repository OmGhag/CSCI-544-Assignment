# CSCI544 HW3 - Named Entity Recognition

## Overview

This assignment implements three NER models on the CoNLL-2003 dataset:
- **Task 1**: Simple Bidirectional LSTM (BLSTM)
- **Task 2**: BLSTM with GloVe embeddings + capitalization features
- **Bonus**: BLSTM with GloVe + capitalization + CNN character-level features

---

## Requirements

```bash
pip install torch numpy seqeval optuna
```

---

## File Structure

```
HW3/
├── data/
│   ├── train
│   ├── dev
│   └── test
├── glove.6B.100d.gz
├── train.py
├── predict.py
├── models.py
├── utils.py
├── optuna_search.py
├── eval/
│   └── eval.py
├── blstm1.pt
├── blstm2.pt
├── blstm_cnn.pt
├── dev1.out
├── dev2.out
├── test1.out
├── test2.out
├── pred
└── README.md
```

---

## Model Architectures

### Task 1 — Simple BLSTM
```
Embedding(100d) → BiLSTM(256d) → Linear(128d) → ELU → Classifier(10)
```

### Task 2 — GloVe + Capitalization BLSTM
```
GloVe Embedding(100d) + Cap Embedding(16d) → BiLSTM(256d) → Linear(128d) → ELU → Classifier(10)
```
Capitalization categories:
- 0: all lowercase (e.g. "london")
- 1: ALL CAPS (e.g. "LONDON")
- 2: Title Case (e.g. "London")
- 3: Mixed/Other (e.g. "iPhone", "1996")

### Bonus — CNN + BLSTM
```
GloVe Embedding(100d) + Cap Embedding(8d) + CNN Char Features(128d)
    → BiLSTM(256d) → Linear(128d) → ELU → Classifier(10)
```

---

## Hyperparameters

### Task 1
| Parameter | Value |
|-----------|-------|
| Embedding dim | 100 |
| LSTM hidden dim | 256 |
| LSTM layers | 1 |
| LSTM dropout | 0.33 |
| Linear output dim | 128 |
| Learning rate | 0.0764 |
| Batch size | 32 |
| Optimizer | SGD (momentum=0.9) |
| Weight decay | 6.88e-4 |
| LR scheduler | ReduceLROnPlateau (patience=3, factor=0.411) |
| Max epochs | 40 |

### Task 2
| Parameter | Value |
|-----------|-------|
| GloVe dim | 100 (fine-tuned) |
| Cap embedding dim | 16 |
| Learning rate | 0.077 |
| Batch size | 32 |
| Weight decay | 5.37e-5 |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.123) |

### Bonus
| Parameter | Value |
|-----------|-------|
| Char embedding dim | 30 |
| CNN filters | 128 |
| CNN kernel size | 5 |
| Cap embedding dim | 8 |
| Learning rate | 0.152 |
| Batch size | 64 |
| Optimizer | Adam |

*Hyperparameters optimized using Optuna (30 trials per task)*

---

## Training

### Task 1 — Simple BLSTM
```bash
python train.py --task 1 \
                --data_dir data \
                --epochs 40 \
                --batch_size 32 \
                --lr 0.0764 \
                --weight_decay 0.000688 \
                --patience 5 \
                --factor 0.411
```

### Task 2 — GloVe + Capitalization
```bash
python train.py --task 2 \
                --data_dir data \
                --glove_path glove.6B.100d.gz \
                --epochs 40 \
                --batch_size 32 \
                --lr 0.077 \
                --weight_decay 0.00005 \
                --patience 5 \
                --factor 0.12
```

### Bonus — CNN Character Model
```bash
python train.py --task bonus \
                --data_dir data \
                --glove_path glove.6B.100d.gz \
                --epochs 40 \
                --batch_size 64 \
                --lr 0.152 \
                --weight_decay 0.000184 \
                --patience 5 \
                --factor 0.41
```

---

## Generating Predictions

### Task 1
```bash
# Dev predictions
python predict.py --task 1 \
                  --model_path blstm1.pt \
                  --data_dir data \
                  --dev_out dev1.out

# Test predictions
python predict.py --task 1 \
                  --model_path blstm1.pt \
                  --data_dir data \
                  --test_out test1.out
```

### Task 2
```bash
# Dev predictions
python predict.py --task 2 \
                  --model_path blstm2.pt \
                  --data_dir data \
                  --glove_path glove.6B.100d.gz \
                  --dev_out dev2.out

# Test predictions
python predict.py --task 2 \
                  --model_path blstm2.pt \
                  --data_dir data \
                  --glove_path glove.6B.100d.gz \
                  --test_out test2.out
```

### Bonus
```bash
# Dev predictions
python predict.py --task bonus \
                  --model_path blstm_cnn.pt \
                  --data_dir data \
                  --glove_path glove.6B.100d.gz \
                  --dev_out dev_cnn.out

# Test predictions (bonus submission file)
python predict.py --task bonus \
                  --model_path blstm_cnn.pt \
                  --data_dir data \
                  --glove_path glove.6B.100d.gz \
                  --test_out pred
```

---

## Evaluation

```bash
python eval/eval.py -p dev1.out -g data/dev
python eval/eval.py -p dev2.out -g data/dev
python eval/eval.py -p dev_cnn.out -g data/dev
```

---

## Results on Dev Set

| Task | Precision | Recall | F1 |
|------|-----------|--------|----|
| Task 1 (BLSTM) | 81.06% | 70.09% | 75.18% |
| Task 2 (GloVe+Cap) | 88.63% | 88.83% | 88.73% |
| Bonus (CNN+BLSTM) | 87.74% | 89.67% | 88.55% |

---

## Hyperparameter Search (Optional)

To run the full Optuna hyperparameter search:

```bash
python optuna_search.py
```

Results are saved to `optuna.db` and can be resumed if interrupted.

---

## Notes

- Random seed fixed at 42 for reproducibility (`torch.manual_seed(42)`)
- All models trained with early stopping based on validation loss
- GloVe embeddings fine-tuned during training (`freeze=False`)
- Unknown words mapped to average GloVe vector
- Capitalization features used to preserve case sensitivity with case-insensitive GloVe