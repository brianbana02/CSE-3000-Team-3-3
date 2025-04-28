import torch
import joblib
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./bert_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('./bert_model')
model.eval()

# Load saved test data
test_texts, test_labels = joblib.load('bert_test_data.pkl')

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

# Create dataset and DataLoader
dataset = HateSpeechDataset(test_texts, test_labels, tokenizer)
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
