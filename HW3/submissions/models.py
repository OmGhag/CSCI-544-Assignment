import torch
import torch.nn as nn

class BLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, 
                 dropout, linear_dim, num_tags):
        super(BLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.blstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                            dropout=dropout, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, linear_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_dim, num_tags)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.blstm(x)
        x = self.linear(x)
        x = self.elu(x)
        x = self.classifier(x)
        return x
    
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for words, tags in loader:
        words = words.to(device)
        tags = tags.to(device)
        # 1. zero gradients
        optimizer.zero_grad()
        # 2. forward pass
        output = model(words)
        # 3. reshape output and tags
        output = output.view(-1, output.size(-1))
        tags = tags.view(-1)
        # 4. compute loss
        loss = criterion(output, tags)
        # 5. backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        # 6. update weights
        optimizer.step()
        # 7. accumulate loss
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_tags = []
    
    with torch.no_grad():
        for words, tags in loader:
            words = words.to(device)
            tags = tags.to(device)
            # forward pass + loss
            output = model(words)
            loss = criterion(output.view(-1, output.size(-1)), tags.view(-1))
            total_loss += loss.item()

            # collect predictions using argmax
            preds = torch.argmax(output, dim=-1)
            
            for pred_seq, tag_seq in zip(preds, tags):
                for p, t in zip(pred_seq, tag_seq):
                    if t != 0:  # ignore padding
                        all_preds.append(p.item())
                        all_tags.append(t.item())


    return total_loss / len(loader), all_preds, all_tags

class BLSTM2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 dropout, linear_dim, num_tags, embedding_matrix, cap_size, cap_embedding_dim):
        super(BLSTM2, self).__init__()
        
        # GloVe embedding
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False, padding_idx=0)
        
        # capitalization embedding
        self.cap_embedding = nn.Embedding(cap_size, cap_embedding_dim)
        
        # BLSTM input dim is now embedding_dim + cap_embedding_dim
        self.blstm = nn.LSTM(embedding_dim + cap_embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        self.linear = nn.Linear(hidden_dim * 2, linear_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_dim, num_tags)
    
    def forward(self, x, cap_ids):
        x = self.embedding(x)
        x = torch.cat((x, self.cap_embedding(cap_ids)), dim=-1)  # concatenate word and cap embeddings
        x, _ = self.blstm(x)
        x = self.linear(x)
        x = self.elu(x)
        x = self.classifier(x)
        return x
    
def train2(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for words, tags, cap in loader:
        words = words.to(device)
        tags = tags.to(device)
        cap = cap.to(device)
        # 1. zero gradients
        optimizer.zero_grad()
        # 2. forward pass
        output = model(words, cap)
        # 3. reshape output and tags
        output = output.view(-1, output.size(-1))
        tags = tags.view(-1)
        # 4. compute loss
        loss = criterion(output, tags)
        # 5. backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        # 6. update weights
        optimizer.step()
        # 7. accumulate loss
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate2(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_tags = []
    
    with torch.no_grad():
        for words, tags ,cap in loader:
            words = words.to(device)
            tags = tags.to(device)
            cap = cap.to(device)
            # forward pass + loss
            output = model(words, cap)
            loss = criterion(output.view(-1, output.size(-1)), tags.view(-1))
            total_loss += loss.item()

            # collect predictions using argmax
            preds = torch.argmax(output, dim=-1)
            
            for pred_seq, tag_seq in zip(preds, tags):
                for p, t in zip(pred_seq, tag_seq):
                    if t != 0:  # ignore padding
                        all_preds.append(p.item())
                        all_tags.append(t.item())


    return total_loss / len(loader), all_preds, all_tags

class CNNBLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 dropout, linear_dim, num_tags, embedding_matrix, cap_size, cap_embedding_dim,
                 char_vocab_size, char_embedding_dim, char_cnn_out_channels, char_cnn_kernel_size):
        super(CNNBLSTM, self).__init__()
        
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False, padding_idx=0)
        self.cap_embedding = nn.Embedding(cap_size, cap_embedding_dim)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        self.char_cnn = nn.Conv1d(in_channels=char_embedding_dim, out_channels=char_cnn_out_channels, kernel_size=char_cnn_kernel_size, padding=char_cnn_kernel_size//2)
        self.blstm = nn.LSTM(embedding_dim + cap_embedding_dim + char_cnn_out_channels, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, linear_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_dim, num_tags)
        
    def forward(self, x, cap_ids, char_ids):
        batch_size, seq_len, max_word_len = char_ids.size()
        char_ids = char_ids.view(-1, max_word_len)  # (batch_size*seq_len, max_word_len)
        char_embeds = self.char_embedding(char_ids)  # (batch*seq_len, max_word_len, char_dim)
        char_embeds = char_embeds.permute(0, 2, 1)   # (batch*seq_len, char_dim, max_word_len)
        char_cnn_out = self.char_cnn(char_embeds)     # (batch*seq_len, num_filters, max_word_len)
        char_cnn_out = torch.max(char_cnn_out, dim=2)[0]  # (batch*seq_len, num_filters)
        char_cnn_out = char_cnn_out.view(batch_size, seq_len, -1)  # (batch_size, seq_len, char_cnn_out_channels)

        x = self.embedding(x)
        cap_embeds = self.cap_embedding(cap_ids)
        x = torch.cat((x, cap_embeds, char_cnn_out), dim=-1)  # concatenate word, cap and char features
        x, _ = self.blstm(x)
        x = self.linear(x)
        x = self.elu(x)
        x = self.classifier(x)
        return x
    
def train_cnn(model_cnn, loader, optimizer, criterion):
    model_cnn.train()
    total_loss = 0
    for words, tags, caps, chars in loader:
        words = words.to(device)
        tags = tags.to(device)
        caps = caps.to(device)
        chars = chars.to(device)
        
        optimizer.zero_grad()
        output = model_cnn(words, caps, chars)
        output = output.view(-1, output.size(-1))
        tags = tags.view(-1)
        loss = criterion(output, tags)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_cnn.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_cnn(model_cnn, loader, criterion):
    model_cnn.eval()
    total_loss = 0
    all_preds = []
    all_tags = []
    
    with torch.no_grad():
        for words, tags, caps, chars in loader:
            words = words.to(device)
            tags = tags.to(device)
            caps = caps.to(device)
            chars = chars.to(device)
            
            output = model_cnn(words, caps, chars)
            loss = criterion(output.view(-1, output.size(-1)), tags.view(-1))
            total_loss += loss.item()

            preds = torch.argmax(output, dim=-1)
            
            for pred_seq, tag_seq in zip(preds, tags):
                for p, t in zip(pred_seq, tag_seq):
                    if t != 0:
                        all_preds.append(p.item())
                        all_tags.append(t.item())

    return total_loss / len(loader), all_preds, all_tags