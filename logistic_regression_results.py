import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Logistic Regression model, TF-IDF vectorizer, and test data
logistic_model = joblib.load("logistic_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
X_test_tfidf, y_test = joblib.load("test_data.pkl")

# Model Predictions
logistic_pred = logistic_model.predict(X_test_tfidf)

# Evaluate Logistic Regression
print("Logistic Regression Performance:")
print(classification_report(y_test, logistic_pred))

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Toxic', 'Toxic'], yticklabels=['Non-Toxic', 'Toxic'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

plot_confusion_matrix(y_test, logistic_pred, "Logistic Regression")
