"""
CSCI 544 - Homework 1
Python Version: 3.12
"""

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import contractions
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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set random seed
RANDOM_STATE = 42

class SentimentAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.vectorizer = CountVectorizer()
        self.models = {
            'Perceptron': Perceptron(random_state=RANDOM_STATE, max_iter=100000),
            'SVM': LinearSVC(random_state=RANDOM_STATE, max_iter=100000),
            'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=100000),
            'Naive Bayes': MultinomialNB()
        }
        self.lemmatizer = WordNetLemmatizer()
        
    def load_data(self):
        self.df = pd.read_csv(self.data_path, sep='\t', on_bad_lines='skip', compression='gzip', low_memory=False)
        self.df = self.df[['review_body', 'star_rating']]
        self.df['star_rating'] = pd.to_numeric(self.df['star_rating'], errors='coerce')
        self.df.dropna(subset=['review_body', 'star_rating'], inplace=True)
        
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
        self.df = pd.concat([positive_df, negative_df]).reset_index(drop=True)
        
        # Print required output
        print(f"Positive reviews: {positive_count}")
        print(f"Negative reviews: {negative_count}")
        print(f"Neutral reviews: {neutral_count}")

    def preprocess_text(self, text):
        """Clean a single text review"""
        text = text.lower()
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = contractions.fix(text)
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
        """Lemmatize text with POS tagging"""
        words = text.split()
        pos_tags = pos_tag(words)
        lemmatized = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos)) 
                      for word, pos in pos_tags]
        return ' '.join(lemmatized)

    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        stop_words = set(stopwords.words('english'))
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
        
    def extract_bigrams(self, text):
        """Extract bigrams from text"""
        words = text.split()
        if len(words) < 2:
            return ""
        bigram_list = list(bigrams(words))
        bigram_strings = ['_'.join(bigram) for bigram in bigram_list]
        return ' '.join(bigram_strings)

    def vectorize_text(self):
        # Extract bigrams
        self.df['bigrams'] = self.df['review_body'].apply(self.extract_bigrams)
        
        # Vectorize
        X = self.vectorizer.fit_transform(self.df['bigrams'])
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
            
            # Print results in required format
            print(f"{model_name} Training Accuracy: {train_acc:.4f}")
            print(f"{model_name} Training Precision: {train_prec:.4f}")
            print(f"{model_name} Training Recall: {train_rec:.4f}")
            print(f"{model_name} Training F1-score: {train_f1:.4f}")
            print(f"{model_name} Testing Accuracy: {test_acc:.4f}")
            print(f"{model_name} Testing Precision: {test_prec:.4f}")
            print(f"{model_name} Testing Recall: {test_rec:.4f}")
            print(f"{model_name} Testing F1-score: {test_f1:.4f}")

if __name__ == "__main__":
    analyzer = SentimentAnalyzer(r'data/amazon_reviews_us_Office_Products_v1_00.tsv.gz')
    analyzer.load_data()
    analyzer.relabel_ratings()
    analyzer.clean_and_preprocess()
    X, y = analyzer.vectorize_text()
    analyzer.train_and_evaluate(X, y)