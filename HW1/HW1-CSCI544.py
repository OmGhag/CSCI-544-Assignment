"""
CSCI 544 - Homework 1
Python Version: 3.12
"""

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, bigrams

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set random seed
RANDOM_STATE = 42

CONTRACTIONS_MAP = {
    "ain't": "is not",
    "amn't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "daren't": "dare not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "everyone's": "everyone is",
    "gimme": "give me",
    "gonna": "going to",
    "gotta": "got to",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "innit": "is it not",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "kinda": "kind of",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "ne'er": "never",
    "o'clock": "of the clock",
    "o'er": "over",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "outta": "out of",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "somebody's": "somebody is",
    "someone's": "someone is",
    "something's": "something is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "tis": "it is",
    "twas": "it was",
    "to've": "to have",
    "wanna": "want to",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "whatcha": "what are you",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who'll've": "who will have",
    "who're": "who are",
    "who's": "who is",
    "why's": "why is",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

class SentimentAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.vectorizer = DictVectorizer(sparse=True)
        self.models = {
            'Perceptron': Perceptron(random_state=RANDOM_STATE, max_iter=10000),
            'SVM': LinearSVC(random_state=RANDOM_STATE, max_iter=10000, C=0.1),
            'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            'Naive Bayes': MultinomialNB()
        }
        self.lemmatizer = WordNetLemmatizer()

    def remove_contractions(self, text):
        # Sort contractions by length (longest first) to handle compound contractions
        contractions_sorted = sorted(CONTRACTIONS_MAP.keys(), key=len, reverse=True)

        # Build pattern with word boundaries
        pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in contractions_sorted) + r')\b',
                            flags=re.IGNORECASE)

        def expand_match(contraction):
            match = contraction.group(0)
            match_lower = match.lower()

            if match_lower in CONTRACTIONS_MAP:
                expanded = CONTRACTIONS_MAP[match_lower]

                # Preserve original capitalization
                if match[0].isupper():
                    expanded = expanded[0].upper() + expanded[1:]

                return expanded

            return match

        # Keep expanding until no more contractions found
        prev_text = ""
        while prev_text != text:
            prev_text = text
            text = pattern.sub(expand_match, text)

        return text

    def load_data(self):
        self.df = pd.read_csv(self.data_path, sep='\t', on_bad_lines='skip', low_memory=False)
        self.df['star_rating'] = pd.to_numeric(self.df['star_rating'], errors='coerce')
        self.df.dropna(subset=['review_body', 'star_rating'], inplace=True)
        self.df = self.df[['review_body', 'star_rating']]

    def relabel_ratings(self):
        # Count neutral reviews before removing
        neutral_count = (self.df['star_rating'] == 3).sum()

        # Remove neutral reviews (rating == 3)
        self.df = self.df[self.df['star_rating'] != 3].copy()

        # Create binary sentiment labels
        self.df['sentiment'] = np.where(self.df['star_rating'] > 3, 1, 0)

        # Count after labeling
        positive_count = (self.df['sentiment'] == 1).sum()
        negative_count = (self.df['sentiment'] == 0).sum()

        # Sample 100k from each class
        positive_df = self.df[self.df['sentiment'] == 1].sample(n=100000, random_state=RANDOM_STATE)
        negative_df = self.df[self.df['sentiment'] == 0].sample(n=100000, random_state=RANDOM_STATE)
        self.df = pd.concat([positive_df, negative_df])

        # Print required output
        print(f"Positive reviews: {positive_count}")
        print(f"Negative reviews: {negative_count}")
        print(f"Neutral reviews: {neutral_count}")

    def preprocess_text(self, text):
        """Clean a single text review"""
        text = text.lower()
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = self.remove_contractions(text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_wordnet_pos(self, treebank_tag):
        """Convert treebank POS tags to WordNet POS tags"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize_text(self, text):
        """Lemmatize text with POS tagging and verb fallback"""
        words = text.split()

        if not words:
            return ""

        pos_tags = pos_tag(words)

        lemmatized = []
        for word, pos in pos_tags:
            primary_pos = self.get_wordnet_pos(pos)
            lemmatized_word = self.lemmatizer.lemmatize(word, primary_pos)

            # If word didn't change and it might be a verb, try verb lemmatization
            if lemmatized_word == word and primary_pos != wordnet.VERB:
                verb_form = self.lemmatizer.lemmatize(word, wordnet.VERB)
                if verb_form != word:
                    lemmatized_word = verb_form

            lemmatized.append(lemmatized_word)

        return ' '.join(lemmatized)

    def remove_stopwords(self, text):
        """Remove stopwords but keep negation words"""
        stop_words = set(stopwords.words('english'))

        # CRITICAL: Keep negation words for sentiment analysis
        negations = {
            'no', 'not', 'nor', 'never', 'neither', 'nobody', 'nothing',
            'nowhere', 'none', 'hardly', 'scarcely', 'barely'
        }
        # Remove negation words from stopwords list
        stop_words = stop_words - negations

        words = text.split()
        filtered = [word for word in words if word not in stop_words]
        return ' '.join(filtered)

    def clean_and_preprocess(self):
        """Apply all cleaning and preprocessing steps"""
        # Calculate average length before cleaning
        avg_before_cleaning = self.df['review_body'].str.len().mean()

        # Apply cleaning
        self.df['review_body'] = self.df['review_body'].apply(self.preprocess_text)

        # Calculate average length after cleaning
        avg_after_cleaning = self.df['review_body'].str.len().mean()

        # Print cleaning results
        print(f"Average length before cleaning: {avg_before_cleaning:.4f}")
        print(f"Average length after cleaning: {avg_after_cleaning:.4f}")

        # Calculate average length before preprocessing
        avg_before_preprocessing = self.df['review_body'].str.len().mean()

        # Remove stopwords
        self.df['review_body'] = self.df['review_body'].apply(self.remove_stopwords)

        # Lemmatize
        self.df['review_body'] = self.df['review_body'].apply(self.lemmatize_text)

        # Calculate average length after preprocessing
        avg_after_preprocessing = self.df['review_body'].str.len().mean()

        # Print preprocessing results
        print(f"Average length before preprocessing: {avg_before_preprocessing:.4f}")
        print(f"Average length after preprocessing: {avg_after_preprocessing:.4f}")

    def bigram_features(self, text):
        """Extract bigrams as binary dictionary with tuple keys"""
        words = text.split()
        if len(words) < 2:
            return {}
        bigram_list = bigrams(words)
        return {bg: 1 for bg in bigram_list}

    def vectorize_text(self):
        # Extract bigram feature dicts
        self.df['bigrams_tuple'] = self.df['review_body'].apply(self.bigram_features)

        # Vectorize using DictVectorizer
        X = self.vectorizer.fit_transform(self.df['bigrams_tuple'])
        y = self.df['sentiment'].values

        return X, y

    def train_and_evaluate(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            train_prec = precision_score(y_train, y_train_pred)
            train_rec = recall_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)

            test_acc = accuracy_score(y_test, y_test_pred)
            test_prec = precision_score(y_test, y_test_pred)
            test_rec = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)

            # Print results
            print(f"{model_name} Training Accuracy: {train_acc:.4f}")
            print(f"{model_name} Training Precision: {train_prec:.4f}")
            print(f"{model_name} Training Recall: {train_rec:.4f}")
            print(f"{model_name} Training F1-score: {train_f1:.4f}")
            print(f"{model_name} Testing Accuracy: {test_acc:.4f}")
            print(f"{model_name} Testing Precision: {test_prec:.4f}")
            print(f"{model_name} Testing Recall: {test_rec:.4f}")
            print(f"{model_name} Testing F1-score: {test_f1:.4f}")

if __name__ == "__main__":
    analyzer = SentimentAnalyzer(r'data.tsv')
    analyzer.load_data()
    analyzer.relabel_ratings()
    analyzer.clean_and_preprocess()
    X, y = analyzer.vectorize_text()
    analyzer.train_and_evaluate(X, y)
