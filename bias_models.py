import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import ssl

# Fix SSL issue for downloading NLTK data
ssl._create_default_https_context = ssl._create_unverified_context

# Download NLTK stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def load_data(data_path):
    """Load text and labels from the dataset."""
    annotations = pd.read_csv(os.path.join(data_path, 'annotations_metadata.csv'))
    
    texts = []
    labels = []
    for _, row in annotations.iterrows():
        file_path = os.path.join(data_path, 'all_files', f"{row['file_id']}.txt")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read().strip())
                labels.append(row['label'])  # Label: 'hate', 'noHate', or other classes
    
    return pd.DataFrame({'text': texts, 'label': labels})

def preprocess_text(text):
    """Cleans the input text."""
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])  # Remove stopwords
    return text

# Load dataset
dataset_path = '/Users/brianbanaszczyk/Downloads/CSE 3000/hate-speech-dataset-master'  # Change this to your local dataset directory
df = load_data(dataset_path)

# Preprocess text
df['text'] = df['text'].apply(preprocess_text)

def encode_labels(label):
    if label == 'hate':
        return 1  # Toxic
    elif label == 'noHate':
        return 0  # Non-toxic
    else:
        return -1  # Unknown/other classes (if applicable)
df['label'] = df['label'].apply(encode_labels)

# Remove unknown labels
df = df[df['label'] != -1]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Text Vectorization using TF-IDF and CountVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

count_vectorizer = CountVectorizer(max_features=5000)
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

# Train a Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_tfidf, y_train)

# Train a Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_count, y_train)

# Predictions
logistic_pred = logistic_model.predict(X_test_tfidf)
nb_pred = nb_model.predict(X_test_count)

# Evaluate Performance
print("Logistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, logistic_pred))
print("Classification Report:\n", classification_report(y_test, logistic_pred))

print("Naive Bayes Performance:")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print("Classification Report:\n", classification_report(y_test, nb_pred))

# Save Models and Vectorizers
joblib.dump(logistic_model, 'logistic_model.pkl')
joblib.dump(nb_model, 'naive_bayes_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(count_vectorizer, 'count_vectorizer.pkl')

print("Models and vectorizers saved successfully.")