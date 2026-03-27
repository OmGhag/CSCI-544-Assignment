import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from models import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['1', '2', 'bonus'])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--glove_path', type=str, default='glove.6B.100d.gz')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dev_out', type=str, default=None)
    parser.add_argument('--test_out', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # common data loading
    train_sentences = read_data(f'{args.data_dir}/train')
    dev_sentences = read_data(f'{args.data_dir}/dev')
    test_sentences = read_data(f'{args.data_dir}/test')
    train_sentences = [s for s in train_sentences if s[0][0] != '-DOCSTART-']
    dev_sentences = [s for s in dev_sentences if s[0][0] != '-DOCSTART-']
    test_sentences = [s for s in test_sentences if s[0][0] != '-DOCSTART-']
    
    # raw for writing predictions
    dev_sentences_raw = read_data(f'{args.data_dir}/dev')
    test_sentences_raw = read_data(f'{args.data_dir}/test')
    
    word2idx, tag2idx = build_vocab(train_sentences)
    idx2tag = {v: k for k, v in tag2idx.items()}
    
    dev_words, dev_tags = encode_data(dev_sentences, word2idx, tag2idx)
    test_words, test_tags = encode_data(test_sentences, word2idx, tag2idx)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    if args.task == '1':
        dev_dataset = NERDataset(dev_words, dev_tags)
        test_dataset = NERDataset(test_words, test_tags)
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
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        eval_fn = lambda loader: evaluate(model, loader, criterion)
    
    elif args.task == '2':
        embeddings = load_glove(args.glove_path, word2idx)
        
        dev_caps = encode_cap_features(dev_sentences)
        test_caps = encode_cap_features(test_sentences)
        
        dev_dataset = NERDataset2(dev_words, dev_tags, dev_caps)
        test_dataset = NERDataset2(test_words, test_tags, test_caps)
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
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        eval_fn = lambda loader: evaluate2(model, loader, criterion)
    
    elif args.task == 'bonus':
        embeddings = load_glove(args.glove_path, word2idx)
        
        dev_caps = encode_cap_features(dev_sentences)
        test_caps = encode_cap_features(test_sentences)
        
        char2idx = build_char_vocab(train_sentences)
        dev_chars = encode_char_data(dev_sentences, char2idx)
        test_chars = encode_char_data(test_sentences, char2idx)
        
        dev_dataset = NERDatasetCNN(dev_words, dev_tags, dev_caps, dev_chars)
        test_dataset = NERDatasetCNN(test_words, test_tags, test_caps, test_chars)
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
            char_cnn_out_channels=128,
            char_cnn_kernel_size=5
        ).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        eval_fn = lambda loader: evaluate_cnn(model, loader, criterion)
    
    # write predictions
    if args.dev_out:
        _, dev_preds, _ = eval_fn(dev_loader)
        write_predictions(dev_sentences_raw, dev_preds, idx2tag, args.dev_out)
        print(f"Dev predictions written to {args.dev_out}")
    
    if args.test_out:
        _, test_preds, _ = eval_fn(test_loader)
        write_predictions(test_sentences_raw, test_preds, idx2tag, args.test_out)
        print(f"Test predictions written to {args.test_out}")
    
    if not args.dev_out and not args.test_out:
        print("No output file specified! Use --dev_out or --test_out")

if __name__ == '__main__':
    main()