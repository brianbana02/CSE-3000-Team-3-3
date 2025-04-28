import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split  # ✅ Added missing import
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Load dataset
dataset_path = 'hate-speech-dataset-master'
annotations_path = os.path.join(dataset_path, 'annotations_metadata.csv')
all_files_path = os.path.join(dataset_path, 'all_files')

annotations = pd.read_csv(annotations_path)
annotations = annotations[annotations['label'].isin(['hate', 'noHate'])]
annotations['label'] = annotations['label'].map({'noHate': 0, 'hate': 1})

def load_text(file_id):
    file_path = os.path.join(all_files_path, f"{file_id}.txt")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    return ""

annotations['text'] = annotations['file_id'].apply(load_text)
annotations = annotations.dropna(subset=['text'])

train_texts, test_texts, train_labels, test_labels = train_test_split(
    annotations['text'].tolist(),
    annotations['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=annotations['label']
)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

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

train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer)
test_dataset = HateSpeechDataset(test_texts, test_labels, tokenizer)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Simpler TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_steps=500,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# Save model
model.save_pretrained('./bert_model')
tokenizer.save_pretrained('./bert_model')

print("DistilBERT model and tokenizer saved successfully.")
