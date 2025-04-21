import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load the trained models and vectorizers
logistic_model = joblib.load("logistic_model.pkl")
nb_model = joblib.load("naive_bayes_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
count_vectorizer = joblib.load("count_vectorizer.pkl")

# Download stopwords if not already downloaded
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

def preprocess_text(text):
    """Clean and preprocess input text."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = " ".join([word for word in text.split() if word not in STOPWORDS])  # Remove stopwords
    return text

def predict_hate_speech(text):
    """Classifies text as hate or non-hate using trained models."""
    text = preprocess_text(text)

    # Convert text into vectorized form
    tfidf_features = tfidf_vectorizer.transform([text])
    count_features = count_vectorizer.transform([text])

    # Predict using both models
    logistic_pred = logistic_model.predict(tfidf_features)[0]
    nb_pred = nb_model.predict(count_features)[0]

    # Convert numerical labels back to human-readable format
    label_map = {0: "Non-Toxic", 1: "Toxic"}
    
    print("\n**Prediction Results**")
    print(f"Logistic Regression Prediction: {label_map[logistic_pred]}")
    print(f"Naive Bayes Prediction: {label_map[nb_pred]}")

# Example Usage
while True:
    text_input = input("\nEnter text to classify (or type 'exit' to stop): ")
    if text_input.lower() == "exit":
        break
    predict_hate_speech(text_input)