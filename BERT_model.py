import os
import torch
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AdamW
import evaluate

# Set dataset path
dataset_path = "/Users/brianbanaszczyk/Downloads/CSE 3000/hate-speech-dataset-master"
all_files_path = os.path.join(dataset_path, "all_files")
annotations_path = os.path.join(dataset_path, "annotations_metadata.csv")

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

# Split dataset into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Save the model
model.save_pretrained("distilbert_hate_speech_model")
tokenizer.save_pretrained("distilbert_hate_speech_model")

print("Model training and evaluation complete. Model saved successfully.")