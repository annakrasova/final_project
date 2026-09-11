import os
import joblib
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from data import data_func, split_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def text_to_sequence(text, word_to_index):
    words = re.findall(r"\b\w+\b", str(text).lower())
    return [word_to_index.get(word, 1) for word in words]

def pad_sequences(sequences, max_length):
    result = np.zeros(
        (len(sequences), max_length),
        dtype=np.int64
    )
    for i, sequence in enumerate(sequences):
        sequence = sequence[:max_length]
        result[i, :len(sequence)] = sequence
    return result

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]
        return self.fc(hidden)

def predict_lstm(model, X_test, word_to_index, max_length):
    sequences = [text_to_sequence(text, word_to_index) for text in X_test]
    sequences = pad_sequences(sequences, max_length)

    X_tensor = torch.tensor(sequences, dtype=torch.long).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1)
    return predictions.cpu().numpy()

def print_metrics(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(model_name)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    return accuracy, macro_f1

def save_confusion_matrix(y_true, y_pred, classes, filename, title):
    matrix = confusion_matrix(y_true, y_pred, labels = classes)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    display.plot(ax=ax, xticks_rotation=45, cmap="Blues")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def print_errors(model_name, X_test, y_test, y_pred, number_of_errors=3):
    wrong_indices = np.where(np.array(y_test) != np.array(y_pred))[0]

    print(f"Ошибочные предсказания: {model_name}")
    if len(wrong_indices) == 0:
        print("Ошибок нет.")
        return

    for number, index in enumerate(wrong_indices[:number_of_errors], start=1):
        print(f"Ошибка {number}")
        print(f"Текст: {X_test.iloc[index]}")
        print(f"Истинный класс: {y_test.iloc[index]}")
        print(f"Предсказанный класс: {y_pred[index]}")



def main():
    classes, X, y, df = data_func()
    X_train, X_test, y_train, y_test = split_data(X, y)

    logistic_model = joblib.load("models/logistic_pipeline.pkl")
    logistic_predictions = logistic_model.predict(X_test)
    logistic_classes = logistic_model.classes_

    logistic_accuracy, logistic_f1 = print_metrics("TF-IDF + Logistic Regression", y_test, logistic_predictions)
    save_confusion_matrix(y_test, logistic_predictions, logistic_classes, "reports/confusion_matrix_logistic.png", "Confusion Matrix - Logistic Regression")
    print_errors("TF-IDF + Logistic Regression", X_test, y_test, logistic_predictions)



    with open("models/tokenizer.pkl","rb") as file:
        tokenizer = pickle.load(file)
    word_to_index = tokenizer["word_to_index"]
    max_length = tokenizer["max_length"]

    checkpoint = torch.load("models/lstm.pt", map_location=device)
    lstm_classes = checkpoint["classes"]
    lstm_model = LSTMClassifier(vocab_size=checkpoint["vocab_size"], embedding_dim=checkpoint["embedding_dim"], hidden_dim=checkpoint["hidden_dim"], num_classes=checkpoint["num_classes"])
    lstm_model.load_state_dict(checkpoint["model_state_dict"])
    lstm_model = lstm_model.to(device)
    lstm_predictions = predict_lstm(lstm_model, X_test, word_to_index, max_length)
    lstm_predictions = np.array([lstm_classes[index] for index in lstm_predictions])

    print()
    lstm_accuracy, lstm_f1 = print_metrics("LSTM", y_test, lstm_predictions)
    
    save_confusion_matrix(y_test, lstm_predictions, lstm_classes, "reports/confusion_matrix_lstm.png", "Confusion Matrix - LSTM")
    print_errors("LSTM", X_test, y_test, lstm_predictions)

    print()
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print(f"{'Model':<30}"f"{'Accuracy':<12}"f"{'Macro-F1':<12}")
    print(f"{'TF-IDF + Logistic Regression':<30}"f"{logistic_accuracy:<12.4f}"f"{logistic_f1:<12.4f}")
    print(f"{'LSTM':<30}"f"{lstm_accuracy:<12.4f}"f"{lstm_f1:<12.4f}")



if __name__ == "__main__":
    main()