import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from models import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, 
                        choices=['1', '2', 'bonus'],
                        help='Which task to train: 1, 2, or bonus')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--glove_path', type=str, default='glove.6B.100d.gz')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0764)
    parser.add_argument('--weight_decay', type=float, default=0.000688)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--factor', type=float, default=0.3)
    args = parser.parse_args()
    
    # ── common for all tasks ──
    train_sentences = read_data(f'{args.data_dir}/train')
    dev_sentences = read_data(f'{args.data_dir}/dev')
    test_sentences = read_data(f'{args.data_dir}/test')
    train_sentences = [s for s in train_sentences if s[0][0] != '-DOCSTART-']
    dev_sentences = [s for s in dev_sentences if s[0][0] != '-DOCSTART-']
    test_sentences = [s for s in test_sentences if s[0][0] != '-DOCSTART-']
    
    word2idx, tag2idx = build_vocab(train_sentences)
    idx2tag = {v: k for k, v in tag2idx.items()}
    
    train_words, train_tags = encode_data(train_sentences, word2idx, tag2idx)
    dev_words, dev_tags = encode_data(dev_sentences, word2idx, tag2idx)
    test_words, test_tags = encode_data(test_sentences, word2idx, tag2idx)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # ── task specific ──
    if args.task == '1':
            # load only word features
            
            train_dataset = NERDataset(train_words, train_tags)
            dev_dataset = NERDataset(dev_words, dev_tags)
            test_dataset = NERDataset(test_words, test_tags)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
            dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
            
            model = BLSTM(
                vocab_size=len(word2idx),
                embedding_dim=100,
                hidden_dim=256,
                num_layers=1,
                dropout=0.33,
                linear_dim=128,
                num_tags=len(tag2idx)
            ).to(device)
            
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, factor=args.factor)
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            
            best_val_loss = float('inf')
            patience_counter = 0
            patience = args.patience
            num_epochs = args.epochs

            for epoch in range(num_epochs):
                train_loss = train(model, train_loader, optimizer, criterion)
                val_loss, preds, true_tags = evaluate(model, dev_loader, criterion)
                scheduler.step(val_loss)

                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 'blstm.pt')
                    patience_counter = 0
                    print(f"  ✓ Best model saved!")
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{patience})")
                    if patience_counter >= patience:
                        print("Early stopping!")
                        break
                    
            model.load_state_dict(torch.load('blstm.pt'))
            print("Best model loaded!")
            
    elif args.task == '2':
        # additionally load glove, cap features
        embeddings = load_glove(args.glove_path, word2idx, embedding_dim=100)
        
        train_caps = encode_cap_features(train_sentences)
        dev_caps = encode_cap_features(dev_sentences)
        test_caps = encode_cap_features(test_sentences)
        
        train_dataset = NERDataset2(train_words, train_tags, train_caps)
        dev_dataset = NERDataset2(dev_words, dev_tags, dev_caps)
        test_dataset = NERDataset2(test_words, test_tags, test_caps)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn2)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn2)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn2)

        model = BLSTM2(
            vocab_size=len(word2idx),
            embedding_dim=100,
            hidden_dim=256,
            num_layers=1,
            dropout=0.33,
            linear_dim=128,
            num_tags=len(tag2idx),
            embedding_matrix=embeddings,
            cap_size=4,
            cap_embedding_dim=16
        ).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, factor=args.factor)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = args.patience
        num_epochs = args.epochs
        
        for epoch in range(num_epochs):
            train_loss = train2(model, train_loader, optimizer, criterion)
            val_loss, preds, true_tags = evaluate2(model, dev_loader, criterion)
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'blstm2.pt')
                patience_counter = 0
                print(f"  ✓ Best model saved!")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print("Early stopping!")
                    break
        
        model.load_state_dict(torch.load('blstm2.pt'))
        print("Best model loaded!")
        
    elif args.task == 'bonus':
        # additionally load glove, cap features, char features
        embeddings = load_glove(args.glove_path, word2idx, embedding_dim=100)
        
        train_caps = encode_cap_features(train_sentences)
        dev_caps = encode_cap_features(dev_sentences)
        test_caps = encode_cap_features(test_sentences)
        
        char2idx = build_char_vocab(train_sentences)
        train_char = encode_char_data(train_sentences, char2idx)
        dev_char = encode_char_data(dev_sentences, char2idx)
        test_char = encode_char_data(test_sentences, char2idx)
        
        train_dataset = NERDatasetCNN(train_words, train_tags, train_caps, train_char)
        dev_dataset = NERDatasetCNN(dev_words, dev_tags, dev_caps, dev_char)
        test_dataset = NERDatasetCNN(test_words, test_tags, test_caps, test_char)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_cnn)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_cnn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_cnn)
        
        model = CNNBLSTM(
            vocab_size=len(word2idx),
            embedding_dim=100,
            hidden_dim=256,
            num_layers=1,
            dropout=0.33,
            linear_dim=128,
            num_tags=len(tag2idx),
            embedding_matrix=embeddings,
            cap_size=4,
            cap_embedding_dim=8,
            char_vocab_size=len(char2idx),
            char_embedding_dim=30,
            char_cnn_out_channels=120,
            char_cnn_kernel_size=5
         ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, factor=args.factor)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = args.patience
        num_epochs = args.epochs
        
        for epoch in range(num_epochs):
            train_loss = train_cnn(model, train_loader, optimizer, criterion)
            val_loss, preds, true_tags = evaluate_cnn(model, dev_loader, criterion)
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'cnnblstm.pt')
                patience_counter = 0
                print(f"  ✓ Best model saved!")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print("Early stopping!")
                    break
                
        model.load_state_dict(torch.load('cnnblstm.pt'))
        print("Best model loaded!")
        

if __name__ == '__main__':
    main()