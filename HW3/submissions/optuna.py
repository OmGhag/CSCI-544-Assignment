import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
from seqeval.metrics import f1_score as seqeval_f1
optuna.logging.set_verbosity(optuna.logging.WARNING)

from utils import *
from models import *

# ── setup device ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# ── load data ──
train_sentences = read_data('data/train')
dev_sentences = read_data('data/dev')
train_sentences = [s for s in train_sentences if s[0][0] != '-DOCSTART-']
dev_sentences = [s for s in dev_sentences if s[0][0] != '-DOCSTART-']

word2idx, tag2idx = build_vocab(train_sentences)
idx2tag = {v: k for k, v in tag2idx.items()}

train_words, train_tags = encode_data(train_sentences, word2idx, tag2idx)
dev_words, dev_tags = encode_data(dev_sentences, word2idx, tag2idx)

# ── task 2 & bonus features ──
embedding_matrix = load_glove('glove.6B.100d.gz', word2idx)
train_caps = encode_cap_features(train_sentences)
dev_caps = encode_cap_features(dev_sentences)
char2idx = build_char_vocab(train_sentences)
train_chars = encode_char_data(train_sentences, char2idx)
dev_chars = encode_char_data(dev_sentences, char2idx)

# ── datasets ──
train_dataset = NERDataset(train_words, train_tags)
dev_dataset = NERDataset(dev_words, dev_tags)

train_dataset2 = NERDataset2(train_words, train_tags, train_caps)
dev_dataset2 = NERDataset2(dev_words, dev_tags, dev_caps)

train_dataset_cnn = NERDatasetCNN(train_words, train_tags, train_caps, train_chars)
dev_dataset_cnn = NERDatasetCNN(dev_words, dev_tags, dev_caps, dev_chars)

# ── evaluate_f1 ──
def evaluate_f1(model, loader, criterion, idx2tag, task='task1'):
    model.eval()
    total_loss = 0
    all_preds = []
    all_tags = []
    
    with torch.no_grad():
        for batch in loader:
            if task == 'task1':
                words, tags = batch
                words, tags = words.to(device), tags.to(device)
                output = model(words)
            elif task == 'task2':
                words, tags, caps = batch
                words, tags, caps = words.to(device), tags.to(device), caps.to(device)
                output = model(words, caps)
            else:  # bonus
                words, tags, caps, chars = batch
                words, tags, caps, chars = words.to(device), tags.to(device), caps.to(device), chars.to(device)
                output = model(words, caps, chars)
            
            loss = criterion(output.view(-1, output.size(-1)), tags.view(-1))
            total_loss += loss.item()
            preds = torch.argmax(output, dim=-1)
            
            for pred_seq, tag_seq in zip(preds, tags):
                pred_sent = []
                true_sent = []
                for p, t in zip(pred_seq, tag_seq):
                    if t != 0:
                        pred_sent.append(idx2tag[p.item()])
                        true_sent.append(idx2tag[t.item()])
                all_preds.append(pred_sent)
                all_tags.append(true_sent)
    
    f1 = seqeval_f1(all_tags, all_preds)
    val_loss = total_loss / len(loader)
    return val_loss, f1

# ============================================================
# TASK 1 OBJECTIVE
# ============================================================
def objective_task1(trial):
    lr = trial.suggest_float('lr', 0.01, 0.3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    patience = trial.suggest_int('patience', 3, 20)
    factor = trial.suggest_float('factor', 0.3, 0.9)

    train_loader_t = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader_t = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model_t = BLSTM(
        vocab_size=len(word2idx),
        embedding_dim=100,
        hidden_dim=256,
        num_layers=1,
        dropout=0.33,
        linear_dim=128,
        num_tags=len(tag2idx)
    ).to(device)

    optimizer_t = torch.optim.SGD(model_t.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_t, patience=patience, factor=factor)
    criterion_t = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = float('inf')
    best_f1 = 0
    patience_counter = 0
    early_stop = 5

    for epoch in range(40):
        train(model_t, train_loader_t, optimizer_t, criterion_t)
        val_loss, f1 = evaluate_f1(model_t, dev_loader_t, criterion_t, idx2tag, task='task1')
        scheduler_t.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_f1 = f1
            patience_counter = 0
            torch.save(model_t.state_dict(), f'optuna_task1_trial{trial.number}.pt')
        else:
            patience_counter += 1
            if patience_counter >= early_stop:
                break

    return best_f1

# ============================================================
# TASK 2 OBJECTIVE
# ============================================================
def objective_task2(trial):
    lr = trial.suggest_float('lr', 1e-3, 3e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    patience = trial.suggest_int('patience', 2, 20)
    factor = trial.suggest_float('factor', 0.1, 0.9)
    cap_embedding_dim = trial.suggest_categorical('cap_embedding_dim', [8, 16, 32])

    train_loader_t = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True, collate_fn=collate_fn2)
    dev_loader_t = DataLoader(dev_dataset2, batch_size=batch_size, shuffle=False, collate_fn=collate_fn2)

    model_t = BLSTM2(
        vocab_size=len(word2idx),
        embedding_dim=100,
        hidden_dim=256,
        num_layers=1,
        dropout=0.33,
        linear_dim=128,
        num_tags=len(tag2idx),
        embedding_matrix=embedding_matrix,
        cap_size=4,
        cap_embedding_dim=cap_embedding_dim
    ).to(device)

    optimizer_t = torch.optim.SGD(model_t.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_t, patience=patience, factor=factor)
    criterion_t = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = float('inf')
    best_f1 = 0
    patience_counter = 0
    early_stop = 5

    for epoch in range(40):
        train2(model_t, train_loader_t, optimizer_t, criterion_t)
        val_loss, f1 = evaluate_f1(model_t, dev_loader_t, criterion_t, idx2tag, task='task2')
        scheduler_t.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_f1 = f1
            patience_counter = 0
            torch.save(model_t.state_dict(), f'optuna_task2_trial{trial.number}.pt')
        else:
            patience_counter += 1
            if patience_counter >= early_stop:
                break

    return best_f1

# ============================================================
# BONUS OBJECTIVE
# ============================================================
def objective_bonus(trial):
    lr = trial.suggest_float('lr', 1e-3, 2e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    patience = trial.suggest_int('patience', 2, 8)
    factor = trial.suggest_float('factor', 0.1, 0.5)
    cap_embedding_dim = trial.suggest_categorical('cap_embedding_dim', [8, 16, 32])
    char_cnn_out_channels = trial.suggest_categorical('char_cnn_out_channels', [64, 100, 128])
    char_cnn_kernel_size = trial.suggest_categorical('char_cnn_kernel_size', [3, 5])

    train_loader_t = DataLoader(train_dataset_cnn, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_cnn)
    dev_loader_t = DataLoader(dev_dataset_cnn, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_cnn)

    model_t = CNNBLSTM(
        vocab_size=len(word2idx),
        embedding_dim=100,
        hidden_dim=256,
        num_layers=1,
        dropout=0.33,
        linear_dim=128,
        num_tags=len(tag2idx),
        embedding_matrix=embedding_matrix,
        cap_size=4,
        cap_embedding_dim=cap_embedding_dim,
        char_vocab_size=len(char2idx),
        char_embedding_dim=30,
        char_cnn_out_channels=char_cnn_out_channels,
        char_cnn_kernel_size=char_cnn_kernel_size
    ).to(device)

    optimizer_t = torch.optim.SGD(model_t.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_t, patience=patience, factor=factor)
    criterion_t = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = float('inf')
    best_f1 = 0
    patience_counter = 0
    early_stop = 5

    for epoch in range(40):
        train_cnn(model_t, train_loader_t, optimizer_t, criterion_t)
        val_loss, f1 = evaluate_f1(model_t, dev_loader_t, criterion_t, idx2tag, task='bonus')
        scheduler_t.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_f1 = f1
            patience_counter = 0
            torch.save(model_t.state_dict(), f'optuna_bonus_trial{trial.number}.pt')
        else:
            patience_counter += 1
            if patience_counter >= early_stop:
                break

    return best_f1

# ============================================================
# RUN ALL THREE STUDIES
# ============================================================
if __name__ == '__main__':
    print("=" * 50)
    print("Running Task 1 study...")
    study1 = optuna.create_study(
        direction='maximize',
        study_name='task1',
        storage='sqlite:///optuna.db',
        load_if_exists=True
    )
    study1.optimize(objective_task1, n_trials=30)
    print(f"Task 1 Best F1: {study1.best_value:.4f}")
    print(f"Task 1 Best params: {study1.best_params}")

    print("=" * 50)
    print("Running Task 2 study...")
    study2 = optuna.create_study(
        direction='maximize',
        study_name='task2',
        storage='sqlite:///optuna.db',
        load_if_exists=True
    )
    study2.optimize(objective_task2, n_trials=30)
    print(f"Task 2 Best F1: {study2.best_value:.4f}")
    print(f"Task 2 Best params: {study2.best_params}")

    print("=" * 50)
    print("Running Bonus study...")
    study3 = optuna.create_study(
        direction='maximize',
        study_name='bonus',
        storage='sqlite:///optuna.db',
        load_if_exists=True
    )
    study3.optimize(objective_bonus, n_trials=30)
    print(f"Bonus Best F1: {study3.best_value:.4f}")
    print(f"Bonus Best params: {study3.best_params}")
