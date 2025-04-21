import os
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load trained model
model_path = "distilbert_hate_speech_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Ensure model is in evaluation mode
model.eval()

# Load dataset and tokenize
import pandas as pd
df = pd.read_csv("/Users/brianbanaszczyk/Downloads/CSE 3000/hate-speech-dataset-master/annotations_metadata.csv")
df = df[df['label'].isin(['hate', 'noHate'])]
df['label'] = df['label'].map({'noHate': 0, 'hate': 1})

# Load text content from files
def load_text(file_id):
    file_path = f"/Users/brianbanaszczyk/Downloads/CSE 3000/hate-speech-dataset-master/all_files/{file_id}.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    return ""

df['text'] = df['file_id'].apply(load_text)

# Tokenize
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()

# Run predictions
y_true = df['label'].tolist()
y_pred = [predict(text) for text in df['text'].tolist()]

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Toxic", "Toxic"], yticklabels=["Non-Toxic", "Toxic"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - DistilBERT")
plt.show()

# Print Classification Report
print("DistilBERT Performance:")
print(classification_report(y_true, y_pred))