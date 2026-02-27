"""
CSCI 544 - Homework 2
Neural Networks for Sentiment Analysis
Python Version: 3.13.9
Library: PyTorch
"""

import pandas as pd
import numpy as np
import re
import gc
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")
import multiprocessing

# NLTK
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Gensim for Word2Vec
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
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Contractions map (same as HW1)
# ---------------------------------------------------------------------------
CONTRACTIONS_MAP = {
    "ain't": "is not", "amn't": "am not", "aren't": "are not",
    "can't": "cannot", "can't've": "cannot have", "'cause": "because",
    "could've": "could have", "couldn't": "could not",
    "couldn't've": "could not have", "daren't": "dare not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "everyone's": "everyone is", "gimme": "give me", "gonna": "going to",
    "gotta": "got to", "hadn't": "had not", "hadn't've": "had not have",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would",
    "he'd've": "he would have", "he'll": "he will",
    "he'll've": "he will have", "he's": "he is", "how'd": "how did",
    "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "i'd": "I would", "i'd've": "I would have", "i'll": "I will",
    "i'll've": "I will have", "i'm": "I am", "i've": "I have",
    "innit": "is it not", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will",
    "it'll've": "it will have", "it's": "it is", "kinda": "kind of",
    "let's": "let us", "ma'am": "madam", "mayn't": "may not",
    "might've": "might have", "mightn't": "might not",
    "mightn't've": "might not have", "must've": "must have",
    "mustn't": "must not", "mustn't've": "must not have",
    "needn't": "need not", "needn't've": "need not have",
    "ne'er": "never", "o'clock": "of the clock", "o'er": "over",
    "oughtn't": "ought not", "oughtn't've": "ought not have",
    "outta": "out of", "shan't": "shall not", "sha'n't": "shall not",
    "shan't've": "shall not have", "she'd": "she would",
    "she'd've": "she would have", "she'll": "she will",
    "she'll've": "she will have", "she's": "she is",
    "should've": "should have", "shouldn't": "should not",
    "shouldn't've": "should not have", "so've": "so have",
    "so's": "so is", "somebody's": "somebody is",
    "someone's": "someone is", "something's": "something is",
    "that'd": "that would", "that'd've": "that would have",
    "that's": "that is", "there'd": "there would",
    "there'd've": "there would have", "there's": "there is",
    "they'd": "they would", "they'd've": "they would have",
    "they'll": "they will", "they'll've": "they will have",
    "they're": "they are", "they've": "they have",
    "tis": "it is", "twas": "it was", "to've": "to have",
    "wanna": "want to", "wasn't": "was not", "we'd": "we would",
    "we'd've": "we would have", "we'll": "we will",
    "we'll've": "we will have", "we're": "we are", "we've": "we have",
    "weren't": "were not", "whatcha": "what are you",
    "what'll": "what will", "what'll've": "what will have",
    "what're": "what are", "what's": "what is", "what've": "what have",
    "when's": "when is", "where'd": "where did", "where's": "where is",
    "who'll": "who will", "who'll've": "who will have",
    "who're": "who are", "who's": "who is", "why's": "why is",
    "won't": "will not", "won't've": "will not have",
    "would've": "would have", "wouldn't": "would not",
    "wouldn't've": "would not have", "y'all": "you all",
    "y'all'd": "you all would", "y'all'd've": "you all would have",
    "y'all're": "you all are", "y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have",
    "you'll": "you will", "you'll've": "you will have",
    "you're": "you are", "you've": "you have",
}

# ---------------------------------------------------------------------------
# Text preprocessing (same pipeline as HW1)
# ---------------------------------------------------------------------------

def remove_contractions(text):
    contractions_sorted = sorted(CONTRACTIONS_MAP.keys(), key=len, reverse=True)
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(k) for k in contractions_sorted) + r')\b',
        flags=re.IGNORECASE,
    )

    def expand_match(m):
        match = m.group(0)
        expanded = CONTRACTIONS_MAP.get(match.lower(), match)
        if match[0].isupper():
            expanded = expanded[0].upper() + expanded[1:]
        return expanded

    prev = ""
    while prev != text:
        prev = text
        text = pattern.sub(expand_match, text)
    return text


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_with_pos(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    if not words:
        return ""
    pos_tags = pos_tag(words)
    lemmatized = []
    for word, pos in pos_tags:
        primary_pos = get_wordnet_pos(pos)
        lemmatized_word = lemmatizer.lemmatize(word, primary_pos)
        if lemmatized_word == word and primary_pos != wordnet.VERB:
            verb_form = lemmatizer.lemmatize(word, wordnet.VERB)
            if verb_form != word:
                lemmatized_word = verb_form
        lemmatized.append(lemmatized_word)
    return ' '.join(lemmatized)


def remove_stopwords_text(text):
    stop_words = set(stopwords.words('english'))
    negations = {
        'no', 'not', 'nor', 'never', 'neither', 'nobody', 'nothing',
        'nowhere', 'none', 'hardly', 'scarcely', 'barely',
    }
    stop_words -= negations
    words = text.split()
    return ' '.join(w for w in words if w not in stop_words)


def preprocess_text(text):
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = remove_contractions(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------------------------------------------------------
# Word2Vec feature helpers
# ---------------------------------------------------------------------------

def get_average_word2vec(review, model, is_custom=False):
    """Average Word2Vec vector for all words in a review."""
    words = review.split()
    vectors = []
    for word in words:
        try:
            vectors.append(model.wv[word] if is_custom else model[word])
        except KeyError:
            continue
    if vectors:
        return np.mean(vectors, axis=0)
    vec_size = model.wv.vector_size if is_custom else model.vector_size
    return np.zeros(vec_size)


def get_concatenated_word2vec(review, model, is_custom=False, max_words=10):
    """Concatenate first max_words Word2Vec vectors, pad with zeros if needed."""
    vec_size = model.wv.vector_size if is_custom else model.vector_size
    words = review.split()[:max_words]
    vectors = []
    for word in words:
        try:
            vectors.append(model.wv[word] if is_custom else model[word])
        except KeyError:
            vectors.append(np.zeros(vec_size))
    while len(vectors) < max_words:
        vectors.append(np.zeros(vec_size))
    return np.concatenate(vectors)


def reviews_to_sequences(reviews, model, max_length=50, is_custom=True):
    """Convert reviews to (num_reviews, max_length, embed_dim) float32 array."""
    vec_size = model.wv.vector_size if is_custom else model.vector_size
    num_reviews = len(reviews)
    sequences = np.zeros((num_reviews, max_length, vec_size), dtype=np.float32)
    for idx, review in enumerate(reviews):
        words = review.split()[:max_length]
        for word_idx, word in enumerate(words):
            try:
                vec = model.wv[word] if is_custom else model[word]
                sequences[idx, word_idx] = vec.astype(np.float32)
            except KeyError:
                pass
    return sequences

# ---------------------------------------------------------------------------
# PyTorch datasets
# ---------------------------------------------------------------------------

class ReviewDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CNNReviewDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ---------------------------------------------------------------------------
# Neural network architectures
# ---------------------------------------------------------------------------

class FeedForwardNN(nn.Module):
    """2-hidden-layer MLP: input -> 50 -> 10 -> output_size."""

    def __init__(self, input_size, output_size, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


class TextCNN(nn.Module):
    """2-layer CNN for text classification with output channels 50 and 10."""

    def __init__(self, embed_dim=300, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(embed_dim, 50, kernel_size=4, padding=1)
        self.conv2 = nn.Conv1d(50, 10, kernel_size=4, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(10, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = self.pool(x).squeeze(2)
        return self.fc(x)

# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_model(model, train_loader, test_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    final_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)

        model.eval()
        correct = total = 0
        running_test_loss = 0.0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                running_test_loss += criterion(outputs, labels).item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        final_accuracy = correct / total
        print(
            f'Epoch [{epoch+1}/{num_epochs}], '
            f'Train Loss: {running_loss / len(train_loader.dataset):.4f}, '
            f'Test Loss: {running_test_loss / len(test_loader.dataset):.4f}, '
            f'Test Accuracy: {final_accuracy:.4f}'
        )
    return final_accuracy

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------------
    # Question 1: Dataset Generation
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Question 1: Dataset Generation")
    print("=" * 60)

    df = pd.read_csv(
        r'C:\Users\omgha\OneDrive\Documents\GitHub\CSCI-544-Assignment\HW2\data\amazon_reviews_us_Office_Products_v1_00.tsv.gz',
        sep='\t', on_bad_lines='skip', low_memory=False,
    )
    df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce')
    df.dropna(subset=['review_body', 'star_rating'], inplace=True)
    df = df[['review_body', 'star_rating']]

    # Sample 50K per rating
    balanced_dfs = []
    for rating in [1, 2, 3, 4, 5]:
        rating_df = df[df['star_rating'] == rating]
        if len(rating_df) >= 50000:
            sampled = rating_df.sample(n=50000, random_state=RANDOM_STATE)
        else:
            print(f"Warning: Only {len(rating_df)} reviews available for rating {rating}")
            sampled = rating_df
        balanced_dfs.append(sampled)
        print(f"Rating {rating}: {len(sampled)} reviews sampled")

    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    print(f"Total balanced dataset size: {df_balanced.shape}")

    # Ternary labels: >3 -> 1 (Positive), <3 -> 2 (Negative), ==3 -> 3 (Neutral)
    def create_ternary_label(rating):
        if rating > 3:
            return 1
        elif rating < 3:
            return 2
        return 3

    df_balanced['label'] = df_balanced['star_rating'].apply(create_ternary_label)
    print("Label distribution:")
    print(df_balanced['label'].value_counts().sort_index())

    del df
    gc.collect()

    # -----------------------------------------------------------------------
    # Data Cleaning
    # -----------------------------------------------------------------------
    print("\nData Cleaning...")
    avg_length_before = df_balanced['review_body'].str.len().mean()
    print(f"Average length before cleaning: {avg_length_before:.4f}")

    df_balanced['review_body'] = df_balanced['review_body'].apply(preprocess_text)

    avg_length_after = df_balanced['review_body'].str.len().mean()
    print(f"Average length after cleaning: {avg_length_after:.4f}")

    # -----------------------------------------------------------------------
    # Preprocessing (stopword removal + lemmatization)
    # -----------------------------------------------------------------------
    print("\nPreprocessing...")
    avg_before_preprocessing = df_balanced['review_body'].str.len().mean()
    print(f"Average length before preprocessing: {avg_before_preprocessing:.4f}")

    df_balanced['review_body'] = df_balanced['review_body'].apply(remove_stopwords_text)
    df_balanced['review_body'] = df_balanced['review_body'].apply(lemmatize_with_pos)

    avg_after_preprocessing = df_balanced['review_body'].str.len().mean()
    print(f"Average length after preprocessing: {avg_after_preprocessing:.4f}")

    # -----------------------------------------------------------------------
    # Question 2: Word Embeddings
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Question 2: Word Embeddings")
    print("=" * 60)

    # 2(a) Load pretrained Word2Vec
    print("\n2(a) Loading pretrained word2vec-google-news-300...")
    pretrained_w2v = api.load('word2vec-google-news-300')
    print(f"Vocabulary size: {len(pretrained_w2v.key_to_index):,}")
    print(f"Vector dimensionality: {pretrained_w2v.vector_size}")

    # Semantic similarity tests
    print("\nSemantic Similarity Test 1: King - Man + Woman")
    try:
        result = pretrained_w2v.most_similar(positive=['king', 'woman'], negative=['man'], topn=5)
        print("Top 5 results:")
        for word, score in result:
            print(f"  {word:15} similarity: {score:.4f}")
    except KeyError as e:
        print(f"Error: {e}")

    print("\nSemantic Similarity Test 2: excellent ~ outstanding")
    try:
        similarity = pretrained_w2v.similarity('excellent', 'outstanding')
        print(f"Similarity(excellent, outstanding) = {similarity:.4f}")
        print("Words most similar to 'excellent':")
        for word, score in pretrained_w2v.most_similar('excellent', topn=5):
            print(f"  {word:15} similarity: {score:.4f}")
    except KeyError as e:
        print(f"Error: {e}")

    # 2(b) Train custom Word2Vec
    print("\n2(b) Training custom Word2Vec model...")
    tokenized_reviews = [review.split() for review in df_balanced['review_body']]
    print(f"Number of reviews: {len(tokenized_reviews):,}")

    custom_w2v = Word2Vec(
        sentences=tokenized_reviews,
        vector_size=300,
        window=11,
        min_count=10,
        workers=multiprocessing.cpu_count(),
        seed=RANDOM_STATE,
        epochs=10,
        sg=0,
        negative=5,
    )
    print(f"Custom vocabulary size: {len(custom_w2v.wv.key_to_index):,}")
    custom_w2v.save('custom_word2vec.model')
    print("Model saved to 'custom_word2vec.model'")

    # Same similarity tests on custom model
    print("\nCustom Model - Semantic Similarity Test 1: King - Man + Woman")
    try:
        result = custom_w2v.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=5)
        print("Top 5 results:")
        for word, score in result:
            print(f"  {word:15} similarity: {score:.4f}")
    except KeyError as e:
        print(f"Error: {e}")

    print("\nCustom Model - Semantic Similarity Test 2: excellent ~ outstanding")
    try:
        similarity = custom_w2v.wv.similarity('excellent', 'outstanding')
        print(f"Similarity(excellent, outstanding) = {similarity:.4f}")
    except KeyError as e:
        print(f"Error: {e}")

    del tokenized_reviews
    gc.collect()

    # -----------------------------------------------------------------------
    # Question 3: Simple Models (Perceptron + SVM) — binary only
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Question 3: Simple Models (Perceptron + SVM)")
    print("=" * 60)

    df_binary = df_balanced[df_balanced['label'].isin([1, 2])].copy()

    print("Generating average Word2Vec vectors for binary dataset...")
    df_binary['pretrained_vec'] = df_binary['review_body'].apply(
        lambda r: get_average_word2vec(r, pretrained_w2v, is_custom=False)
    )
    df_binary['custom_vec'] = df_binary['review_body'].apply(
        lambda r: get_average_word2vec(r, custom_w2v, is_custom=True)
    )

    x_pretrained_bin = np.vstack(df_binary['pretrained_vec'])
    x_custom_bin = np.vstack(df_binary['custom_vec'])
    y_bin = df_binary['label'].values

    x_train_pre, x_test_pre, y_train, y_test = train_test_split(
        x_pretrained_bin, y_bin, test_size=0.2, random_state=RANDOM_STATE
    )
    x_train_cust, x_test_cust, _, _ = train_test_split(
        x_custom_bin, y_bin, test_size=0.2, random_state=RANDOM_STATE
    )

    # Perceptron + Pretrained
    perc_pre = Perceptron(random_state=RANDOM_STATE, max_iter=1000)
    perc_pre.fit(x_train_pre, y_train)
    acc_perc_pre = accuracy_score(perc_pre.predict(x_test_pre), y_test)
    print(f"Perceptron + Pretrained W2V Testing Accuracy: {acc_perc_pre:.4f}")

    # Perceptron + Custom
    perc_cust = Perceptron(random_state=RANDOM_STATE, max_iter=1000)
    perc_cust.fit(x_train_cust, y_train)
    acc_perc_cust = accuracy_score(perc_cust.predict(x_test_cust), y_test)
    print(f"Perceptron + Custom W2V Testing Accuracy: {acc_perc_cust:.4f}")

    # SVM + Pretrained
    svm_pre = LinearSVC(random_state=RANDOM_STATE, max_iter=1000, C=0.1)
    svm_pre.fit(x_train_pre, y_train)
    acc_svm_pre = accuracy_score(svm_pre.predict(x_test_pre), y_test)
    print(f"SVM + Pretrained W2V Testing Accuracy: {acc_svm_pre:.4f}")

    # SVM + Custom
    svm_cust = LinearSVC(random_state=RANDOM_STATE, max_iter=1000, C=0.1)
    svm_cust.fit(x_train_cust, y_train)
    acc_svm_cust = accuracy_score(svm_cust.predict(x_test_cust), y_test)
    print(f"SVM + Custom W2V Testing Accuracy: {acc_svm_cust:.4f}")

    del perc_pre, perc_cust, svm_pre, svm_cust
    del x_pretrained_bin, x_custom_bin
    gc.collect()

    # Also compute averaged vectors for all 3 classes (ternary) for Q4
    print("\nGenerating average Word2Vec vectors for ternary dataset...")
    df_balanced['pretrained_vec'] = df_balanced['review_body'].apply(
        lambda r: get_average_word2vec(r, pretrained_w2v, is_custom=False)
    )
    df_balanced['custom_vec'] = df_balanced['review_body'].apply(
        lambda r: get_average_word2vec(r, custom_w2v, is_custom=True)
    )

    # -----------------------------------------------------------------------
    # Question 4(a): FFNN with averaged vectors
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Question 4(a): FFNN with Averaged Vectors")
    print("=" * 60)

    # ---- Binary ----
    df_binary_avg = df_balanced[df_balanced['label'].isin([1, 2])].copy()
    x_pre_bin = np.vstack(df_binary_avg['pretrained_vec'])
    x_cust_bin = np.vstack(df_binary_avg['custom_vec'])
    y_bin_avg = df_binary_avg['label'].values - 1  # 0-indexed

    x_tr_pre_bin, x_te_pre_bin, y_tr_bin, y_te_bin = train_test_split(
        x_pre_bin, y_bin_avg, test_size=0.2, random_state=RANDOM_STATE
    )
    x_tr_cust_bin, x_te_cust_bin, _, _ = train_test_split(
        x_cust_bin, y_bin_avg, test_size=0.2, random_state=RANDOM_STATE
    )

    print("\n4(a) Binary - Pretrained W2V:")
    torch.manual_seed(RANDOM_STATE)
    model_bin_pre = FeedForwardNN(300, 2)
    train_loader = DataLoader(ReviewDataset(x_tr_pre_bin, y_tr_bin), batch_size=64, shuffle=True)
    test_loader = DataLoader(ReviewDataset(x_te_pre_bin, y_te_bin), batch_size=64, shuffle=False)
    acc_ffnn_avg_pre_bin = train_model(model_bin_pre, train_loader, test_loader)
    print(f"FFNN Avg + Pretrained W2V Binary Testing Accuracy: {acc_ffnn_avg_pre_bin:.4f}")

    print("\n4(a) Binary - Custom W2V:")
    torch.manual_seed(RANDOM_STATE)
    model_bin_cust = FeedForwardNN(300, 2)
    train_loader = DataLoader(ReviewDataset(x_tr_cust_bin, y_tr_bin), batch_size=64, shuffle=True)
    test_loader = DataLoader(ReviewDataset(x_te_cust_bin, y_te_bin), batch_size=64, shuffle=False)
    acc_ffnn_avg_cust_bin = train_model(model_bin_cust, train_loader, test_loader)
    print(f"FFNN Avg + Custom W2V Binary Testing Accuracy: {acc_ffnn_avg_cust_bin:.4f}")

    del x_pre_bin, x_cust_bin, x_tr_pre_bin, x_te_pre_bin, x_tr_cust_bin, x_te_cust_bin
    del df_binary_avg
    gc.collect()

    # ---- Ternary ----
    x_pre_ter = np.vstack(df_balanced['pretrained_vec'])
    x_cust_ter = np.vstack(df_balanced['custom_vec'])
    y_ter = df_balanced['label'].values - 1  # 0-indexed

    x_tr_pre_ter, x_te_pre_ter, y_tr_ter, y_te_ter = train_test_split(
        x_pre_ter, y_ter, test_size=0.2, random_state=RANDOM_STATE
    )
    x_tr_cust_ter, x_te_cust_ter, _, _ = train_test_split(
        x_cust_ter, y_ter, test_size=0.2, random_state=RANDOM_STATE
    )

    print("\n4(a) Ternary - Pretrained W2V:")
    torch.manual_seed(RANDOM_STATE)
    model_ter_pre = FeedForwardNN(300, 3)
    train_loader = DataLoader(ReviewDataset(x_tr_pre_ter, y_tr_ter), batch_size=64, shuffle=True)
    test_loader = DataLoader(ReviewDataset(x_te_pre_ter, y_te_ter), batch_size=64, shuffle=False)
    acc_ffnn_avg_pre_ter = train_model(model_ter_pre, train_loader, test_loader)
    print(f"FFNN Avg + Pretrained W2V Ternary Testing Accuracy: {acc_ffnn_avg_pre_ter:.4f}")

    print("\n4(a) Ternary - Custom W2V:")
    torch.manual_seed(RANDOM_STATE)
    model_ter_cust = FeedForwardNN(300, 3)
    train_loader = DataLoader(ReviewDataset(x_tr_cust_ter, y_tr_ter), batch_size=64, shuffle=True)
    test_loader = DataLoader(ReviewDataset(x_te_cust_ter, y_te_ter), batch_size=64, shuffle=False)
    acc_ffnn_avg_cust_ter = train_model(model_ter_cust, train_loader, test_loader)
    print(f"FFNN Avg + Custom W2V Ternary Testing Accuracy: {acc_ffnn_avg_cust_ter:.4f}")

    del x_pre_ter, x_cust_ter, x_tr_pre_ter, x_te_pre_ter, x_tr_cust_ter, x_te_cust_ter
    gc.collect()

    # -----------------------------------------------------------------------
    # Question 4(b): FFNN with concatenated first 10 vectors (3000-dim)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Question 4(b): FFNN with Concatenated First 10 Vectors")
    print("=" * 60)

    def build_concat_features(df_subset, model, is_custom, label_col='label'):
        reviews = df_subset['review_body'].tolist()
        X = np.array([
            get_concatenated_word2vec(r, model, is_custom=is_custom) for r in reviews
        ])
        y = df_subset[label_col].values - 1
        return X, y

    # ---- Binary ----
    df_bin_concat = df_balanced[df_balanced['label'].isin([1, 2])].copy()

    print("\nGenerating binary concatenated features (pretrained)...")
    X_bin_pre, y_bin_c = build_concat_features(df_bin_concat, pretrained_w2v, is_custom=False)
    X_tr_bin_pre, X_te_bin_pre, y_tr_bc, y_te_bc = train_test_split(
        X_bin_pre, y_bin_c, test_size=0.2, random_state=RANDOM_STATE
    )
    del X_bin_pre

    print("\nGenerating binary concatenated features (custom)...")
    X_bin_cust, _ = build_concat_features(df_bin_concat, custom_w2v, is_custom=True)
    X_tr_bin_cust, X_te_bin_cust, _, _ = train_test_split(
        X_bin_cust, y_bin_c, test_size=0.2, random_state=RANDOM_STATE
    )
    del X_bin_cust
    del df_bin_concat
    gc.collect()

    print("\n4(b) Binary - Pretrained W2V:")
    torch.manual_seed(RANDOM_STATE)
    model_bin_concat_pre = FeedForwardNN(3000, 2)
    train_loader = DataLoader(ReviewDataset(X_tr_bin_pre, y_tr_bc), batch_size=64, shuffle=True)
    test_loader = DataLoader(ReviewDataset(X_te_bin_pre, y_te_bc), batch_size=64, shuffle=False)
    acc_ffnn_concat_pre_bin = train_model(model_bin_concat_pre, train_loader, test_loader)
    print(f"FFNN Concat + Pretrained W2V Binary Testing Accuracy: {acc_ffnn_concat_pre_bin:.4f}")
    del X_tr_bin_pre, X_te_bin_pre

    print("\n4(b) Binary - Custom W2V:")
    torch.manual_seed(RANDOM_STATE)
    model_bin_concat_cust = FeedForwardNN(3000, 2)
    train_loader = DataLoader(ReviewDataset(X_tr_bin_cust, y_tr_bc), batch_size=64, shuffle=True)
    test_loader = DataLoader(ReviewDataset(X_te_bin_cust, y_te_bc), batch_size=64, shuffle=False)
    acc_ffnn_concat_cust_bin = train_model(model_bin_concat_cust, train_loader, test_loader)
    print(f"FFNN Concat + Custom W2V Binary Testing Accuracy: {acc_ffnn_concat_cust_bin:.4f}")
    del X_tr_bin_cust, X_te_bin_cust
    gc.collect()

    # ---- Ternary ----
    print("\nGenerating ternary concatenated features (pretrained)...")
    X_ter_pre, y_ter_c = build_concat_features(df_balanced, pretrained_w2v, is_custom=False)
    X_tr_ter_pre, X_te_ter_pre, y_tr_tc, y_te_tc = train_test_split(
        X_ter_pre, y_ter_c, test_size=0.2, random_state=RANDOM_STATE
    )
    del X_ter_pre

    print("\nGenerating ternary concatenated features (custom)...")
    X_ter_cust, _ = build_concat_features(df_balanced, custom_w2v, is_custom=True)
    X_tr_ter_cust, X_te_ter_cust, _, _ = train_test_split(
        X_ter_cust, y_ter_c, test_size=0.2, random_state=RANDOM_STATE
    )
    del X_ter_cust
    gc.collect()

    print("\n4(b) Ternary - Pretrained W2V:")
    torch.manual_seed(RANDOM_STATE)
    model_ter_concat_pre = FeedForwardNN(3000, 3)
    train_loader = DataLoader(ReviewDataset(X_tr_ter_pre, y_tr_tc), batch_size=64, shuffle=True)
    test_loader = DataLoader(ReviewDataset(X_te_ter_pre, y_te_tc), batch_size=64, shuffle=False)
    acc_ffnn_concat_pre_ter = train_model(model_ter_concat_pre, train_loader, test_loader)
    print(f"FFNN Concat + Pretrained W2V Ternary Testing Accuracy: {acc_ffnn_concat_pre_ter:.4f}")
    del X_tr_ter_pre, X_te_ter_pre

    print("\n4(b) Ternary - Custom W2V:")
    torch.manual_seed(RANDOM_STATE)
    model_ter_concat_cust = FeedForwardNN(3000, 3)
    train_loader = DataLoader(ReviewDataset(X_tr_ter_cust, y_tr_tc), batch_size=64, shuffle=True)
    test_loader = DataLoader(ReviewDataset(X_te_ter_cust, y_te_tc), batch_size=64, shuffle=False)
    acc_ffnn_concat_cust_ter = train_model(model_ter_concat_cust, train_loader, test_loader)
    print(f"FFNN Concat + Custom W2V Ternary Testing Accuracy: {acc_ffnn_concat_cust_ter:.4f}")
    del X_tr_ter_cust, X_te_ter_cust
    gc.collect()

    # -----------------------------------------------------------------------
    # Question 5: Convolutional Neural Networks
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Question 5: Convolutional Neural Networks")
    print("=" * 60)

    # ---- Binary ----
    df_bin_cnn = df_balanced[df_balanced['label'].isin([1, 2])].copy()
    train_idx_bin, test_idx_bin = train_test_split(
        range(len(df_bin_cnn)), test_size=0.2, random_state=RANDOM_STATE
    )

    # Pretrained binary
    print("\nQ5 Binary CNN - Pretrained W2V:")
    print("Generating train sequences (pretrained)...")
    X_tr_cnn_pre_bin = reviews_to_sequences(
        df_bin_cnn.iloc[train_idx_bin]['review_body'], pretrained_w2v,
        max_length=50, is_custom=False
    )
    print("Generating test sequences (pretrained)...")
    X_te_cnn_pre_bin = reviews_to_sequences(
        df_bin_cnn.iloc[test_idx_bin]['review_body'], pretrained_w2v,
        max_length=50, is_custom=False
    )
    y_tr_cnn_bin = df_bin_cnn.iloc[train_idx_bin]['label'].values - 1
    y_te_cnn_bin = df_bin_cnn.iloc[test_idx_bin]['label'].values - 1

    torch.manual_seed(RANDOM_STATE)
    model_cnn_bin_pre = TextCNN(embed_dim=300, num_classes=2)
    train_loader = DataLoader(CNNReviewDataset(X_tr_cnn_pre_bin, y_tr_cnn_bin), batch_size=64, shuffle=True)
    test_loader = DataLoader(CNNReviewDataset(X_te_cnn_pre_bin, y_te_cnn_bin), batch_size=64, shuffle=False)
    acc_cnn_pre_bin = train_model(model_cnn_bin_pre, train_loader, test_loader)
    print(f"CNN + Pretrained W2V Binary Testing Accuracy: {acc_cnn_pre_bin:.4f}")
    del X_tr_cnn_pre_bin, X_te_cnn_pre_bin
    gc.collect()

    # Custom binary
    print("\nQ5 Binary CNN - Custom W2V:")
    print("Generating train sequences (custom)...")
    X_tr_cnn_cust_bin = reviews_to_sequences(
        df_bin_cnn.iloc[train_idx_bin]['review_body'], custom_w2v,
        max_length=50, is_custom=True
    )
    print("Generating test sequences (custom)...")
    X_te_cnn_cust_bin = reviews_to_sequences(
        df_bin_cnn.iloc[test_idx_bin]['review_body'], custom_w2v,
        max_length=50, is_custom=True
    )

    torch.manual_seed(RANDOM_STATE)
    model_cnn_bin_cust = TextCNN(embed_dim=300, num_classes=2)
    train_loader = DataLoader(CNNReviewDataset(X_tr_cnn_cust_bin, y_tr_cnn_bin), batch_size=64, shuffle=True)
    test_loader = DataLoader(CNNReviewDataset(X_te_cnn_cust_bin, y_te_cnn_bin), batch_size=64, shuffle=False)
    acc_cnn_cust_bin = train_model(model_cnn_bin_cust, train_loader, test_loader)
    print(f"CNN + Custom W2V Binary Testing Accuracy: {acc_cnn_cust_bin:.4f}")
    del X_tr_cnn_cust_bin, X_te_cnn_cust_bin, df_bin_cnn
    gc.collect()

    # ---- Ternary ----
    train_idx_ter, test_idx_ter = train_test_split(
        range(len(df_balanced)), test_size=0.2, random_state=RANDOM_STATE
    )

    # Pretrained ternary
    print("\nQ5 Ternary CNN - Pretrained W2V:")
    print("Generating train sequences (pretrained)...")
    X_tr_cnn_pre_ter = reviews_to_sequences(
        df_balanced.iloc[train_idx_ter]['review_body'], pretrained_w2v,
        max_length=50, is_custom=False
    )
    print("Generating test sequences (pretrained)...")
    X_te_cnn_pre_ter = reviews_to_sequences(
        df_balanced.iloc[test_idx_ter]['review_body'], pretrained_w2v,
        max_length=50, is_custom=False
    )
    y_tr_cnn_ter = df_balanced.iloc[train_idx_ter]['label'].values - 1
    y_te_cnn_ter = df_balanced.iloc[test_idx_ter]['label'].values - 1

    torch.manual_seed(RANDOM_STATE)
    model_cnn_ter_pre = TextCNN(embed_dim=300, num_classes=3)
    train_loader = DataLoader(CNNReviewDataset(X_tr_cnn_pre_ter, y_tr_cnn_ter), batch_size=64, shuffle=True)
    test_loader = DataLoader(CNNReviewDataset(X_te_cnn_pre_ter, y_te_cnn_ter), batch_size=64, shuffle=False)
    acc_cnn_pre_ter = train_model(model_cnn_ter_pre, train_loader, test_loader)
    print(f"CNN + Pretrained W2V Ternary Testing Accuracy: {acc_cnn_pre_ter:.4f}")
    del X_tr_cnn_pre_ter, X_te_cnn_pre_ter
    gc.collect()

    # Custom ternary
    print("\nQ5 Ternary CNN - Custom W2V:")
    print("Generating train sequences (custom)...")
    X_tr_cnn_cust_ter = reviews_to_sequences(
        df_balanced.iloc[train_idx_ter]['review_body'], custom_w2v,
        max_length=50, is_custom=True
    )
    print("Generating test sequences (custom)...")
    X_te_cnn_cust_ter = reviews_to_sequences(
        df_balanced.iloc[test_idx_ter]['review_body'], custom_w2v,
        max_length=50, is_custom=True
    )

    torch.manual_seed(RANDOM_STATE)
    model_cnn_ter_cust = TextCNN(embed_dim=300, num_classes=3)
    train_loader = DataLoader(CNNReviewDataset(X_tr_cnn_cust_ter, y_tr_cnn_ter), batch_size=64, shuffle=True)
    test_loader = DataLoader(CNNReviewDataset(X_te_cnn_cust_ter, y_te_cnn_ter), batch_size=64, shuffle=False)
    acc_cnn_cust_ter = train_model(model_cnn_ter_cust, train_loader, test_loader)
    print(f"CNN + Custom W2V Ternary Testing Accuracy: {acc_cnn_cust_ter:.4f}")
    del X_tr_cnn_cust_ter, X_te_cnn_cust_ter
    gc.collect()

    # -----------------------------------------------------------------------
    # Final Summary: All 16 Accuracy Values
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SUMMARY: ALL 16 ACCURACY VALUES")
    print("=" * 60)

    print("\nQ3: Simple Models (Binary Classification)")
    print(f"  Perceptron + Pretrained W2V:            {acc_perc_pre:.4f}")
    print(f"  Perceptron + Custom W2V:                {acc_perc_cust:.4f}")
    print(f"  SVM + Pretrained W2V:                   {acc_svm_pre:.4f}")
    print(f"  SVM + Custom W2V:                       {acc_svm_cust:.4f}")

    print("\nQ4(a): FFNN with Averaged Vectors")
    print(f"  FFNN Avg + Pretrained W2V (Binary):     {acc_ffnn_avg_pre_bin:.4f}")
    print(f"  FFNN Avg + Custom W2V (Binary):         {acc_ffnn_avg_cust_bin:.4f}")
    print(f"  FFNN Avg + Pretrained W2V (Ternary):    {acc_ffnn_avg_pre_ter:.4f}")
    print(f"  FFNN Avg + Custom W2V (Ternary):        {acc_ffnn_avg_cust_ter:.4f}")

    print("\nQ4(b): FFNN with Concatenated First 10 Vectors")
    print(f"  FFNN Concat + Pretrained W2V (Binary):  {acc_ffnn_concat_pre_bin:.4f}")
    print(f"  FFNN Concat + Custom W2V (Binary):      {acc_ffnn_concat_cust_bin:.4f}")
    print(f"  FFNN Concat + Pretrained W2V (Ternary): {acc_ffnn_concat_pre_ter:.4f}")
    print(f"  FFNN Concat + Custom W2V (Ternary):     {acc_ffnn_concat_cust_ter:.4f}")

    print("\nQ5: Convolutional Neural Networks")
    print(f"  CNN + Pretrained W2V (Binary):          {acc_cnn_pre_bin:.4f}")
    print(f"  CNN + Custom W2V (Binary):              {acc_cnn_cust_bin:.4f}")
    print(f"  CNN + Pretrained W2V (Ternary):         {acc_cnn_pre_ter:.4f}")
    print(f"  CNN + Custom W2V (Ternary):             {acc_cnn_cust_ter:.4f}")


if __name__ == "__main__":
    main()
