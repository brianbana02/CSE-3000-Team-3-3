import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./bert_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('./bert_model')
model.eval()

# Set dataset path
dataset_path = 'hate-speech-dataset-master'
annotations_path = os.path.join(dataset_path, 'annotations_metadata.csv')
all_files_path = os.path.join(dataset_path, 'all_files')

# Load dataset
annotations = pd.read_csv(annotations_path)
annotations = annotations[annotations['label'].isin(['hate', 'noHate'])]
annotations['label'] = annotations['label'].map({'noHate': 0, 'hate': 1})

# Load text content
def load_text(file_id):
    file_path = os.path.join(all_files_path, f"{file_id}.txt")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    return ""

annotations['text'] = annotations['file_id'].apply(load_text)
annotations = annotations.dropna(subset=['text'])

texts = annotations['text'].tolist()
labels = annotations['label'].tolist()

# Create dataset class
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = HateSpeechDataset(texts, labels, tokenizer)

# Create DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=16)

# Get predictions
y_preds = []
y_trues = []

for batch in loader:
    with torch.no_grad():
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        y_preds.extend(predictions.cpu().numpy())
        y_trues.extend(batch['labels'].cpu().numpy())

# Evaluate
print("DistilBERT Performance:")
print(classification_report(y_trues, y_preds))

# Confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Toxic', 'Toxic'], yticklabels=['Non-Toxic', 'Toxic'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

plot_confusion_matrix(y_trues, y_preds, "DistilBERT")

# Test model on sample sentences
test_sentences = [
    "I love my culture and my people.",
    "Women should stay at home and not work.",
    "Immigrants are ruining this country.",
    "Black people are amazing and resilient.",
]

def predict_sentences(sentences):
    encodings = tokenizer(sentences, truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encodings)
        preds = torch.argmax(outputs.logits, dim=-1)
    label_map = {0: "Non-Toxic", 1: "Toxic"}
    for sentence, pred in zip(sentences, preds):
        print(f"\nText: {sentence}")
        print(f"Prediction: {label_map[pred.item()]}")

predict_sentences(test_sentences)
