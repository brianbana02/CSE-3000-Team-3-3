import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import ssl

# Fix SSL issue for downloading NLTK data
ssl._create_default_https_context = ssl._create_unverified_context

# Download NLTK stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def load_data(data_path):
    annotations = pd.read_csv(os.path.join(data_path, 'annotations_metadata.csv'))
    texts = []
    labels = []
    for _, row in annotations.iterrows():
        file_path = os.path.join(data_path, 'all_files', f"{row['file_id']}.txt")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read().strip())
                labels.append(row['label'])
    return pd.DataFrame({'text': texts, 'label': labels})

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

# Load dataset
dataset_path = 'hate-speech-dataset-master'
df = load_data(dataset_path)

# Preprocess text
df['text'] = df['text'].apply(preprocess_text)

def encode_labels(label):
    if label == 'hate':
        return 1
    elif label == 'noHate':
        return 0
    else:
        return -1

df['label'] = df['label'].apply(encode_labels)
df = df[df['label'] != -1]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Text Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_tfidf, y_train)

# Predictions
logistic_pred = logistic_model.predict(X_test_tfidf)

# Evaluate Performance
print("Logistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, logistic_pred))
print("Classification Report:\n", classification_report(y_test, logistic_pred))

# Save Model, Vectorizer, and Test Data
joblib.dump(logistic_model, 'logistic_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump((X_test_tfidf, y_test), 'test_data.pkl')

print("Logistic Regression model, TF-IDF vectorizer, and test data saved.")
