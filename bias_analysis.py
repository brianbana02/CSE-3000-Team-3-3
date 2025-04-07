import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set correct dataset path
dataset_path = "hate-speech-dataset-master"
all_files_path = os.path.join(dataset_path, "all_files")
annotations_path = os.path.join(dataset_path, "annotations_metadata.csv")

# Verify dataset path
if not os.path.exists(annotations_path):
    raise FileNotFoundError(f"Metadata file not found at {annotations_path}. Check your dataset path.")

# Load the trained models and vectorizers
logistic_model = joblib.load("logistic_model.pkl")
nb_model = joblib.load("naive_bayes_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
count_vectorizer = joblib.load("count_vectorizer.pkl")

# Load dataset metadata
df = pd.read_csv(annotations_path)
print("Available columns in dataset:", df.columns)

# Filter only relevant labels (hate and noHate)
df = df[df['label'].isin(['hate', 'noHate'])]
df['label'] = df['label'].map({'noHate': 0, 'hate': 1})

# Function to load text content from files
def load_text(file_id):
    file_path = os.path.join(all_files_path, f"{file_id}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    return ""

# Load text data
df['text'] = df['file_id'].apply(load_text)

# Check if text was successfully loaded
if df['text'].isnull().all():
    raise ValueError("No text data found. Ensure the 'all_files' directory is correctly set.")

# Check class distribution
print("Class Distribution:")
print(df['label'].value_counts())

# Load test data
X_test_tfidf = tfidf_vectorizer.transform(df['text'])
X_test_count = count_vectorizer.transform(df['text'])
y_test = df['label']

# Model Predictions
logistic_pred = logistic_model.predict(X_test_tfidf)
nb_pred = nb_model.predict(X_test_count)

# Evaluate False Positives & False Negatives
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Toxic', 'Toxic'], yticklabels=['Non-Toxic', 'Toxic'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

print("Logistic Regression Performance:")
print(classification_report(y_test, logistic_pred))
plot_confusion_matrix(y_test, logistic_pred, "Logistic Regression")

print("Naive Bayes Performance:")
print(classification_report(y_test, nb_pred))
plot_confusion_matrix(y_test, nb_pred, "Naive Bayes")

# Test the model with sample sentences containing different contexts
test_sentences = [
    "I love my culture and my people.",
    "Women should stay at home and not work.",
    "Immigrants are ruining this country.",
    "Black people are amazing and resilient.",
]

def predict_bias(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    text_count = count_vectorizer.transform([text])
    
    logistic_result = logistic_model.predict(text_tfidf)[0]
    nb_result = nb_model.predict(text_count)[0]
    
    label_map = {0: "Non-Toxic", 1: "Toxic"}
    print(f"\nText: {text}")
    print(f"Logistic Regression: {label_map[logistic_result]}")
    print(f"Naive Bayes: {label_map[nb_result]}")

for sentence in test_sentences:
    predict_bias(sentence)
