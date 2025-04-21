import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the trained model and tokenizer
model_path = "distilbert_hate_speech_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Ensure model is in evaluation mode
model.eval()

# Function to predict text class (toxic or non-toxic)
def predict_toxicity(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    label_map = {0: "Non-Toxic", 1: "Toxic"}
    return label_map[predicted_class]

# Test the model with sample inputs
test_sentences = [
    "I love my culture and my people.",
    "Women should stay at home and not work.",
    "Immigrants are ruining this country.",
    "Black people are amazing and resilient.",
]

for sentence in test_sentences:
    print(f"Text: {sentence}")
    print(f"DistilBERT Prediction: {predict_toxicity(sentence)}\n")