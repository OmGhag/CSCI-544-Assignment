#!/usr/bin/env python3
"""
CSCI 544 - Homework 2: Sentiment Analysis
Author: Om Ghatiyali

Simple script to run all homework experiments.
Generates 16 accuracy values across different models.
"""

import pandas as pd
import numpy as np
import re
import gc
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")

# NLTK
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Gensim
import gensim.downloader as api
from gensim.models import Word2Vec

# Scikit-learn
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_PATH = r'/home/omghag/CSCI-544-Assignment/HW2/data/amazon_reviews_us_Office_Products_v1_00.tsv.gz'

# Dataset settings
DATASET_SIZE = 250000
TEST_SIZE = 0.2

# Training settings
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5
NUM_EPOCHS = 10

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

class TextPreprocessor:
    def __init__(self):
        # Download NLTK data
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Contractions dictionary
        self.contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have", "isn't": "is not",
            "it's": "it is", "let's": "let us", "shouldn't": "should not",
            "that's": "that is", "there's": "there is", "they'd": "they would",
            "they'll": "they will", "they're": "they are", "wasn't": "was not",
            "we'd": "we would", "we're": "we are", "weren't": "were not",
            "what's": "what is", "won't": "will not", "wouldn't": "would not",
            "you'd": "you would", "you're": "you are", "you've": "you have"
        }
    
    def get_wordnet_pos(self, tag):
        # Convert POS tag to wordnet format
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def expand_contractions(self, text):
        # Expand contractions
        pattern = re.compile('({})'.format('|'.join(self.contractions.keys())),
                           flags=re.IGNORECASE | re.DOTALL)
        
        def replace(match):
            return self.contractions[match.group(0).lower()]
        
        return pattern.sub(replace, text)
    
    def preprocess(self, text):
        # Remove HTML
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenize and POS tag
        tokens = text.split()
        pos_tags = pos_tag(tokens)
        
        # Lemmatize
        lemmatized = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos))
            for word, pos in pos_tags
        ]
        
        # Remove single characters
        lemmatized = [w for w in lemmatized if len(w) > 1]
        
        return ' '.join(lemmatized)


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class LazyReviewDataset(Dataset):
    # Memory-efficient dataset that generates features on-the-fly
    
    def __init__(self, reviews, labels, model, max_length=50, 
                 is_custom=True, feature_type='sequence'):
        self.reviews = reviews.reset_index(drop=True)
        self.labels = torch.LongTensor(labels)
        self.model = model
        self.max_length = max_length
        self.is_custom = is_custom
        self.feature_type = feature_type
        self.vector_size = model.wv.vector_size if is_custom else model.vector_size
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = self.reviews.iloc[idx]
        label = self.labels[idx]
        
        # Generate features based on type
        if self.feature_type == 'averaged':
            feature = self._to_averaged(review)
        elif self.feature_type == 'sequence':
            feature = self._to_sequence(review)
        else:  # concatenated
            feature = self._to_concatenated(review)
        
        return torch.FloatTensor(feature), label
    
    def _to_averaged(self, review):
        # Average all word vectors
        words = review.split()
        vectors = []
        
        for word in words:
            try:
                vec = self.model.wv[word] if self.is_custom else self.model[word]
                vectors.append(vec.astype(np.float32))
            except KeyError:
                pass
        
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.vector_size, dtype=np.float32)
    
    def _to_sequence(self, review):
        # Create sequence of word vectors
        words = review.split()[:self.max_length]
        sequence = np.zeros((self.max_length, self.vector_size), dtype=np.float32)
        
        for i, word in enumerate(words):
            try:
                vec = self.model.wv[word] if self.is_custom else self.model[word]
                sequence[i] = vec.astype(np.float32)
            except KeyError:
                pass
        
        return sequence
    
    def _to_concatenated(self, review):
        # Concatenate first N word vectors
        words = review.split()[:self.max_length]
        vectors = []
        
        for word in words:
            try:
                vec = self.model.wv[word] if self.is_custom else self.model[word]
                vectors.append(vec.astype(np.float32))
            except KeyError:
                vectors.append(np.zeros(self.vector_size, dtype=np.float32))
        
        # Pad to max_length
        while len(vectors) < self.max_length:
            vectors.append(np.zeros(self.vector_size, dtype=np.float32))
        
        return np.concatenate(vectors)


# ============================================================================
# NEURAL NETWORK MODELS
# ============================================================================

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.5):
        super(FeedForwardNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x


class TextCNN(nn.Module):
    def __init__(self, embed_dim=300, num_classes=2, dropout_rate=0.5):
        super(TextCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(embed_dim, 50, kernel_size=4, padding=1)
        self.conv2 = nn.Conv1d(50, 10, kernel_size=4, padding=1)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(10, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Transpose for conv1d
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.pool(x)
        x = x.squeeze(2)
        x = self.fc(x)
        
        return x


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, test_loader, num_epochs=10):
    device = torch.device(DEVICE)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    return test_acc


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("CSCI 544 - HOMEWORK 2: SENTIMENT ANALYSIS")
    print("="*80)
    print(f"\nDevice: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}\n")
    
    # ========================================================================
    # Q1: LOAD AND PREPROCESS DATASET
    # ========================================================================
    print("="*80)
    print("Q1: LOADING AND PREPROCESSING DATASET")
    print("="*80)
    
    # Load data
    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH, sep='\t', on_bad_lines='skip', low_memory=False)
    print(f"Initial size: {len(df):,}")
    
    # Clean data
    df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce')
    df.dropna(subset=['review_body', 'star_rating'], inplace=True)
    print(f"After cleaning: {len(df):,}")
    
    # Create labels
    def rating_to_label(rating):
        if rating <= 2:
            return 1  # Negative
        elif rating == 3:
            return 2  # Neutral
        else:
            return 3  # Positive
    
    df['label'] = df['star_rating'].apply(rating_to_label)
    
    # Create balanced dataset
    print("\nCreating balanced dataset...")
    samples_per_class = 83334
    df_balanced = pd.concat([
        df[df['label'] == 1].sample(n=samples_per_class, random_state=RANDOM_STATE),
        df[df['label'] == 2].sample(n=samples_per_class, random_state=RANDOM_STATE),
        df[df['label'] == 3].sample(n=samples_per_class, random_state=RANDOM_STATE)
    ], ignore_index=True)
    
    print(f"Balanced dataset size: {len(df_balanced):,}")
    
    # Preprocess text
    print("\nPreprocessing reviews...")
    preprocessor = TextPreprocessor()
    df_balanced['review_body'] = df_balanced['review_body'].apply(
        lambda x: preprocessor.preprocess(x)
    )
    print("Preprocessing complete!")
    
    # ========================================================================
    # Q2: WORD EMBEDDINGS
    # ========================================================================
    print("\n" + "="*80)
    print("Q2: LOADING WORD EMBEDDINGS")
    print("="*80)
    
    # Load pretrained
    print("\nLoading pretrained Word2Vec...")
    pretrained_w2v = api.load('word2vec-google-news-300')
    print(f"Pretrained vocabulary: {len(pretrained_w2v.key_to_index):,}")
    
    # Train custom
    print("\nTraining custom Word2Vec...")
    sentences = [review.split() for review in df_balanced['review_body']]
    custom_w2v = Word2Vec(
        sentences=sentences,
        vector_size=300,
        window=5,
        min_count=5,
        workers=4,
        epochs=10,
        seed=RANDOM_STATE
    )
    print(f"Custom vocabulary: {len(custom_w2v.wv.key_to_index):,}")
    
    # ========================================================================
    # Q3: SIMPLE MODELS
    # ========================================================================
    print("\n" + "="*80)
    print("Q3: SIMPLE MODELS (PERCEPTRON + SVM)")
    print("="*80)
    
    # Prepare binary dataset
    df_binary = df_balanced[df_balanced['label'].isin([1, 2])].copy()
    X_train_reviews, X_test_reviews, y_train, y_test = train_test_split(
        df_binary['review_body'],
        df_binary['label'].values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    # Helper function for sklearn models
    def train_sklearn_model(model, w2v_model, is_custom):
        # Create datasets
        train_dataset = LazyReviewDataset(
            X_train_reviews, y_train, w2v_model,
            is_custom=is_custom, feature_type='averaged'
        )
        test_dataset = LazyReviewDataset(
            X_test_reviews, y_test, w2v_model,
            is_custom=is_custom, feature_type='averaged'
        )
        
        # Generate features
        train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        X_train = []
        for batch_X, _ in train_loader:
            X_train.append(batch_X.numpy())
        X_train = np.vstack(X_train)
        
        X_test = []
        for batch_X, _ in test_loader:
            X_test.append(batch_X.numpy())
        X_test = np.vstack(X_test)
        
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Cleanup
        del X_train, X_test, train_dataset, test_dataset
        gc.collect()
        
        return acc
    
    # Train models
    print("\n1. Perceptron + Pretrained...")
    acc_perc_pre = train_sklearn_model(
        Perceptron(random_state=RANDOM_STATE, max_iter=1000),
        pretrained_w2v, is_custom=False
    )
    print(f"   Accuracy: {acc_perc_pre:.4f}")
    
    print("\n2. Perceptron + Custom...")
    acc_perc_cust = train_sklearn_model(
        Perceptron(random_state=RANDOM_STATE, max_iter=1000),
        custom_w2v, is_custom=True
    )
    print(f"   Accuracy: {acc_perc_cust:.4f}")
    
    print("\n3. SVM + Pretrained...")
    acc_svm_pre = train_sklearn_model(
        LinearSVC(random_state=RANDOM_STATE, max_iter=1000),
        pretrained_w2v, is_custom=False
    )
    print(f"   Accuracy: {acc_svm_pre:.4f}")
    
    print("\n4. SVM + Custom...")
    acc_svm_cust = train_sklearn_model(
        LinearSVC(random_state=RANDOM_STATE, max_iter=1000),
        custom_w2v, is_custom=True
    )
    print(f"   Accuracy: {acc_svm_cust:.4f}")
    
    # Cleanup
    del df_binary, X_train_reviews, X_test_reviews
    gc.collect()
    
    # ========================================================================
    # Q4(a): FFNN WITH AVERAGED FEATURES
    # ========================================================================
    print("\n" + "="*80)
    print("Q4(a): FEEDFORWARD NN WITH AVERAGED FEATURES")
    print("="*80)
    
    # Helper function
    def train_ffnn_averaged(df, w2v_model, is_custom, is_binary):
        # Prepare dataset
        if is_binary:
            df_subset = df[df['label'].isin([1, 2])].copy()
            num_classes = 2
        else:
            df_subset = df[df['label'].isin([1, 2, 3])].copy()
            num_classes = 3
        
        train_idx, test_idx = train_test_split(
            range(len(df_subset)), test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        train_reviews = df_subset.iloc[train_idx]['review_body']
        test_reviews = df_subset.iloc[test_idx]['review_body']
        y_train = df_subset.iloc[train_idx]['label'].values - 1
        y_test = df_subset.iloc[test_idx]['label'].values - 1
        
        # Create datasets
        train_dataset = LazyReviewDataset(
            train_reviews, y_train, w2v_model,
            is_custom=is_custom, feature_type='averaged'
        )
        test_dataset = LazyReviewDataset(
            test_reviews, y_test, w2v_model,
            is_custom=is_custom, feature_type='averaged'
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Train model
        model = FeedForwardNN(input_size=300, output_size=num_classes)
        acc = train_model(model, train_loader, test_loader)
        
        # Cleanup
        del train_dataset, test_dataset, train_loader, test_loader, df_subset
        gc.collect()
        
        return acc
    
    print("\n1. Binary + Pretrained...")
    acc_ffnn_avg_binary_pre = train_ffnn_averaged(
        df_balanced, pretrained_w2v, is_custom=False, is_binary=True
    )
    
    print("\n2. Binary + Custom...")
    acc_ffnn_avg_binary_cust = train_ffnn_averaged(
        df_balanced, custom_w2v, is_custom=True, is_binary=True
    )
    
    print("\n3. Ternary + Pretrained...")
    acc_ffnn_avg_ternary_pre = train_ffnn_averaged(
        df_balanced, pretrained_w2v, is_custom=False, is_binary=False
    )
    
    print("\n4. Ternary + Custom...")
    acc_ffnn_avg_ternary_cust = train_ffnn_averaged(
        df_balanced, custom_w2v, is_custom=True, is_binary=False
    )
    
    # ========================================================================
    # Q4(b): FFNN WITH CONCATENATED FEATURES
    # ========================================================================
    print("\n" + "="*80)
    print("Q4(b): FEEDFORWARD NN WITH CONCATENATED FEATURES")
    print("="*80)
    
    # Helper function
    def train_ffnn_concat(df, w2v_model, is_custom, is_binary):
        # Prepare dataset
        if is_binary:
            df_subset = df[df['label'].isin([1, 2])].copy()
            num_classes = 2
        else:
            df_subset = df[df['label'].isin([1, 2, 3])].copy()
            num_classes = 3
        
        train_idx, test_idx = train_test_split(
            range(len(df_subset)), test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        train_reviews = df_subset.iloc[train_idx]['review_body']
        test_reviews = df_subset.iloc[test_idx]['review_body']
        y_train = df_subset.iloc[train_idx]['label'].values - 1
        y_test = df_subset.iloc[test_idx]['label'].values - 1
        
        # Create datasets
        train_dataset = LazyReviewDataset(
            train_reviews, y_train, w2v_model,
            max_length=10, is_custom=is_custom, feature_type='concatenated'
        )
        test_dataset = LazyReviewDataset(
            test_reviews, y_test, w2v_model,
            max_length=10, is_custom=is_custom, feature_type='concatenated'
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Train model
        model = FeedForwardNN(input_size=3000, output_size=num_classes)
        acc = train_model(model, train_loader, test_loader)
        
        # Cleanup
        del train_dataset, test_dataset, train_loader, test_loader, df_subset
        gc.collect()
        
        return acc
    
    print("\n1. Binary + Pretrained...")
    acc_ffnn_concat_binary_pre = train_ffnn_concat(
        df_balanced, pretrained_w2v, is_custom=False, is_binary=True
    )
    
    print("\n2. Binary + Custom...")
    acc_ffnn_concat_binary_cust = train_ffnn_concat(
        df_balanced, custom_w2v, is_custom=True, is_binary=True
    )
    
    print("\n3. Ternary + Pretrained...")
    acc_ffnn_concat_ternary_pre = train_ffnn_concat(
        df_balanced, pretrained_w2v, is_custom=False, is_binary=False
    )
    
    print("\n4. Ternary + Custom...")
    acc_ffnn_concat_ternary_cust = train_ffnn_concat(
        df_balanced, custom_w2v, is_custom=True, is_binary=False
    )
    
    # ========================================================================
    # Q5: CNN
    # ========================================================================
    print("\n" + "="*80)
    print("Q5: CONVOLUTIONAL NEURAL NETWORKS")
    print("="*80)
    
    # Helper function
    def train_cnn(df, w2v_model, is_custom, is_binary):
        # Prepare dataset
        if is_binary:
            df_subset = df[df['label'].isin([1, 2])].copy()
            num_classes = 2
        else:
            df_subset = df[df['label'].isin([1, 2, 3])].copy()
            num_classes = 3
        
        train_idx, test_idx = train_test_split(
            range(len(df_subset)), test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        train_reviews = df_subset.iloc[train_idx]['review_body']
        test_reviews = df_subset.iloc[test_idx]['review_body']
        y_train = df_subset.iloc[train_idx]['label'].values - 1
        y_test = df_subset.iloc[test_idx]['label'].values - 1
        
        # Create datasets
        train_dataset = LazyReviewDataset(
            train_reviews, y_train, w2v_model,
            max_length=50, is_custom=is_custom, feature_type='sequence'
        )
        test_dataset = LazyReviewDataset(
            test_reviews, y_test, w2v_model,
            max_length=50, is_custom=is_custom, feature_type='sequence'
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Train model
        model = TextCNN(embed_dim=300, num_classes=num_classes)
        acc = train_model(model, train_loader, test_loader)
        
        # Cleanup
        del train_dataset, test_dataset, train_loader, test_loader, df_subset
        gc.collect()
        
        return acc
    
    print("\n1. Binary + Pretrained...")
    acc_cnn_binary_pre = train_cnn(
        df_balanced, pretrained_w2v, is_custom=False, is_binary=True
    )
    
    print("\n2. Binary + Custom...")
    acc_cnn_binary_cust = train_cnn(
        df_balanced, custom_w2v, is_custom=True, is_binary=True
    )
    
    print("\n3. Ternary + Pretrained...")
    acc_cnn_ternary_pre = train_cnn(
        df_balanced, pretrained_w2v, is_custom=False, is_binary=False
    )
    
    print("\n4. Ternary + Custom...")
    acc_cnn_ternary_cust = train_cnn(
        df_balanced, custom_w2v, is_custom=True, is_binary=False
    )
    
    # ========================================================================
    # PRINT FINAL RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS - 16 ACCURACY VALUES")
    print("="*80)
    
    print("\nQ3: SIMPLE MODELS (4 values)")
    print("-" * 80)
    print(f"Model              Pretrained    Custom")
    print(f"Perceptron         {acc_perc_pre:.4f}        {acc_perc_cust:.4f}")
    print(f"SVM                {acc_svm_pre:.4f}        {acc_svm_cust:.4f}")
    
    print("\nQ4(a): FFNN AVERAGED (4 values)")
    print("-" * 80)
    print(f"Classification     Pretrained    Custom")
    print(f"Binary             {acc_ffnn_avg_binary_pre:.4f}        {acc_ffnn_avg_binary_cust:.4f}")
    print(f"Ternary            {acc_ffnn_avg_ternary_pre:.4f}        {acc_ffnn_avg_ternary_cust:.4f}")
    
    print("\nQ4(b): FFNN CONCATENATED (4 values)")
    print("-" * 80)
    print(f"Classification     Pretrained    Custom")
    print(f"Binary             {acc_ffnn_concat_binary_pre:.4f}        {acc_ffnn_concat_binary_cust:.4f}")
    print(f"Ternary            {acc_ffnn_concat_ternary_pre:.4f}        {acc_ffnn_concat_ternary_cust:.4f}")
    
    print("\nQ5: CNN (4 values)")
    print("-" * 80)
    print(f"Classification     Pretrained    Custom")
    print(f"Binary             {acc_cnn_binary_pre:.4f}        {acc_cnn_binary_cust:.4f}")
    print(f"Ternary            {acc_cnn_ternary_pre:.4f}        {acc_cnn_ternary_cust:.4f}")
    
    print("\n" + "="*80)
    print("✅ ALL 16 ACCURACY VALUES COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()