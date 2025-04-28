# CSE 3000 Team 3 - Hate Speech Detection Project

## Overview

This project builds and evaluates two models for hate speech detection:

- Logistic Regression with TF-IDF vectorization
- DistilBERT Model: Lightweight transformer model trained to detect hate speech

The dataset contains 10,703 samples labeled as "hate" or "noHate".  
Data is split 80% for training and 20% for testing.

---

## Features

- **Logistic Regression Model:** Lightweight baseline using TF-IDF and Logistic Regression.
- **DistilBERT Model:** Lightweight transformer model trained to detect hate speech.
- **Data Splitting:** 80/20 train-test split with stratified labels.
- **Model Saving:** Trained models and vectorizers are saved for reuse.
- **Evaluation Metrics:** Accuracy, precision, recall, F1-score, and confusion matrices are generated.

---

## Project Structure

```
CSE-3000-Team-3/
├── bert_model/                 # Saved DistilBERT model and tokenizer
├── hate-speech-dataset-master/  # Dataset (annotations and text files)
├── results/                     # BERT model checkpoints
├── logistic_regression_training.py   # Logistic Regression training script
├── logistic_regression_results.py    # Logistic Regression evaluation script
├── bert_model_training.py       # BERT training script
├── bert_model_results.py        # BERT evaluation script
├── test_data.pkl                # 20% test set for Logistic Regression
├── bert_test_data.pkl           # 20% test set for BERT
├── requirements.txt             # Python package dependencies
└── README.md                    # Project overview (this file)
```

---

## Installation

### Prerequisites

- Python 3.8+
- Pip

### Steps

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd CSE-3000-Team-3
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Models

```bash
# Train and Evaluate Models

# Logistic Regression
python logistic_regression_training.py
python logistic_regression_results.py

# DistilBERT
python bert_model_training.py
python bert_model_results.py
```

---

## Notes

- Trained models and large files are stored directly in this repository using Git LFS.
- Evaluation results include full classification reports and confusion matrices for both models.

---

## Contributors

- Brian Banaszczyk (brb19012)
- Jasmine Yee (jay22005)
- Lakshita Ganesh Kumar (lgk21002)
- Neil Wong (nbw21002)
