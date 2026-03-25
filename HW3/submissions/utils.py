import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import gzip
import numpy as np

def read_data(filepath):
    sentences = [[]]
    for line in open(filepath):
        line = line.strip()
        if line == "":
            sentences.append([])
        else:
            try:
                idx, word, tag = line.split()
            except ValueError:
                idx, word = line.split()
                tag = "O"
            sentences[-1].append((word, tag))
    if sentences[-1] == []:
        sentences.pop()
    return sentences

def build_vocab(sentences):
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    tag2idx = {"<PAD>": 0}
    
    for sentence in sentences:
        for word, tag in sentence:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx)
    
    return word2idx, tag2idx

def encode_data(sentences, word2idx, tag2idx):
    word_ids = [[word2idx.get(word, word2idx["<UNK>"]) for word, tag in sentence] for sentence in sentences]
    tag_ids = [[tag2idx[tag] for word, tag in sentence] for sentence in sentences]
    return word_ids, tag_ids

def encode_cap_features(sentences):
    cap_ids = [[get_cap_feature(word) for word, tag in sentence] 
               for sentence in sentences]
    return cap_ids

def get_cap_feature(word):
    # returns 0, 1, 2, or 3
    # 0 = all lowercase
    if word.islower():
        return 0
    # 1 = all caps
    elif word.isupper():
        return 1
    # 2 = title case
    elif word.istitle():
        return 2
    # 3 = other/mixed
    else:
        return 3

def build_char_vocab(sentences):
    char2idx = {"<PAD>": 0, "<UNK>": 1}
    for sentence in sentences:
        for word, tag in sentence:
            for char in word:
                if char not in char2idx:
                    char2idx[char] = len(char2idx)
    return char2idx

def word_to_char_ids(word, char2idx, max_word_len=20):
    char_ids = [char2idx.get(c, char2idx["<UNK>"]) for c in word[:max_word_len]]
    # pad with zeros if shorter than max_word_len
    char_ids += [char2idx["<PAD>"]] * (max_word_len - len(char_ids))
    return char_ids

def encode_char_data(sentences, char2idx, max_word_len=20):
    # returns list of lists of char_id lists
    # each sentence → list of words → list of char ids
    encoded = []
    for sentence in sentences:
        word_encoded = []
        for word, tag in sentence:
            char_ids = [char2idx.get(c, char2idx['<UNK>']) for c in word[:max_word_len]]
            char_ids += [char2idx['<PAD>']] * (max_word_len - len(char_ids))
            word_encoded.append(char_ids)
        encoded.append(word_encoded)
    return encoded

def load_glove(glove_path, word2idx, embedding_dim=100):
    # step 1: initialize embedding matrix with zeros or random
    embedding_matrix = np.zeros((len(word2idx), embedding_dim))
    
    # step 2: load glove vectors into a dict
    glove_dict = {}
    with gzip.open(glove_path, 'rt', encoding='utf-8') as f:
        for line in f:
            # each line is: "word 0.1 0.2 ... 0.n"
            # parse it here
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            glove_dict[word] = vector
    
    # step 3: compute average vector for UNK
    avg_vector = np.mean(list(glove_dict.values()), axis=0)
    
    # step 4: fill embedding matrix
    for word, idx in word2idx.items():
        # use glove vector if available, else avg_vector
        embedding_matrix[idx] = glove_dict.get(word.lower(), avg_vector)
    
    return embedding_matrix

def write_predictions(sentences, preds, idx2tag, output_file, include_docstart=True):
    with open(output_file, "w") as f:
        flat_idx = 0
        for sentence in sentences:
            if include_docstart and sentence[0][0] == '-DOCSTART-':
                f.write("1 -DOCSTART- O\n\n")
                continue
            for i, (word, _) in enumerate(sentence):
                pred_tag = idx2tag[preds[flat_idx]]
                f.write(f"{i+1} {word} {pred_tag}\n")
                flat_idx += 1
            f.write("\n")
    
    # remove trailing newline
    with open(output_file, 'rb+') as f:
        f.seek(-1, 2)
        f.truncate()

class NERDataset(Dataset):
    def __init__(self, word_ids, tag_ids):
        self.word_ids = word_ids
        self.tag_ids = tag_ids

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.word_ids[idx]), torch.tensor(self.tag_ids[idx])
    
class NERDataset2(Dataset):
    def __init__(self, word_ids, tag_ids, cap_ids):
        self.word_ids = word_ids
        self.tag_ids = tag_ids
        self.cap_ids = cap_ids

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.word_ids[idx]), torch.tensor(self.tag_ids[idx]), torch.tensor(self.cap_ids[idx])
    
class NERDatasetCNN(Dataset):
    def __init__(self, word_ids, tag_ids, cap_ids, char_ids):
        self.word_ids = word_ids
        self.tag_ids = tag_ids
        self.cap_ids = cap_ids
        self.char_ids = char_ids

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, idx):
        return (torch.tensor(self.word_ids[idx]), 
                torch.tensor(self.tag_ids[idx]), 
                torch.tensor(self.cap_ids[idx]), 
                torch.tensor(self.char_ids[idx]))
        
def collate_fn(batch):
    word_ids, tag_ids = zip(*batch)
    word_ids_padded = pad_sequence(word_ids, batch_first=True, padding_value=0)
    tag_ids_padded = pad_sequence(tag_ids, batch_first=True, padding_value=0)
    return word_ids_padded, tag_ids_padded

def collate_fn2(batch):
    word_ids, tag_ids, cap_ids = zip(*batch)
    word_ids_padded = pad_sequence(word_ids, batch_first=True, padding_value=0)
    tag_ids_padded = pad_sequence(tag_ids, batch_first=True, padding_value=0)
    cap_ids_padded = pad_sequence(cap_ids, batch_first=True, padding_value=0)
    return word_ids_padded, tag_ids_padded, cap_ids_padded

def collate_fn_cnn(batch):
    word_ids, tag_ids, cap_ids, char_ids = zip(*batch)
    
    word_ids_padded = pad_sequence(word_ids, batch_first=True, padding_value=0)
    tag_ids_padded = pad_sequence(tag_ids, batch_first=True, padding_value=0)
    cap_ids_padded = pad_sequence(cap_ids, batch_first=True, padding_value=0)
    
    # char_ids: each element is (seq_len, max_word_len)
    # pad_sequence will pad along seq_len dimension automatically!
    char_ids_padded = pad_sequence(char_ids, batch_first=True, padding_value=0)
    # result shape: (batch_size, max_seq_len, max_word_len)
    
    return word_ids_padded, tag_ids_padded, cap_ids_padded, char_ids_padded