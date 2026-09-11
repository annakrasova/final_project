import re
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from data import data_func, split_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_vocab(texts, max_words=20000):
    word_counts = {}
    for text in texts:
        words = re.findall( r"\b\w+\b", str(text).lower())
        for word in words:
            word_counts[word] = (word_counts.get(word, 0) + 1)
    sorted_words = sorted(
        word_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )
    word_to_index = {
        "<PAD>": 0,
        "<UNK>": 1
    }
    for word, _ in sorted_words[:max_words - 2]:
        word_to_index[word] = len(word_to_index)
    return word_to_index

def text_to_sequence(text, word_to_index):
    words = re.findall(r"\b\w+\b", str(text).lower())
    return [word_to_index.get(word, 1) for word in words]

def get_max_length(sequences):
    lengths = [len(sequence) for sequence in sequences]
    max_length = int(np.percentile(lengths, 95))
    return max(1, max_length)

def pad_sequences(sequences, max_length):
    result = np.zeros(
        (len(sequences), max_length),
        dtype=np.int64
    )
    for i, sequence in enumerate(sequences):
        sequence = sequence[:max_length]
        result[i, :len(sequence)] = sequence
    return result

def encode_labels(labels, classes):
    class_to_index = {
    class_name: index
    for index, class_name in enumerate(classes)
    }
    return np.array([class_to_index[label] for label in labels], dtype=np.int64)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, word_to_index, max_length):
        self.sequences = [text_to_sequence(
                text,
                word_to_index)
            for text in texts
        ]

        self.sequences = pad_sequences(self.sequences, max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return (torch.tensor(
                self.sequences[index],
                dtype=torch.long
            ), torch.tensor(
                self.labels[index],
                dtype=torch.long
            )
        )

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

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(
            outputs,
            y_batch
        )
        loss.backward()
        optimizer.step()
        total_loss += (
            loss.item() * X_batch.size(0)
        )
    return total_loss / len(loader.dataset)

def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(
                outputs,
                y_batch
            )
            total_loss += (
                loss.item() * X_batch.size(0)
            )

    return total_loss / len(loader.dataset)

def predict(model, loader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predicted = torch.argmax(outputs, dim=1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(y_batch.numpy())

    return (
        np.array(true_labels),
        np.array(predictions)
    )

def main():
    print(f"Используемое устройство: {device}")

    classes, X, y, df = data_func()
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    word_to_index = build_vocab(X_train, max_words=20000)
    print(f"Размер словаря: {len(word_to_index)}")
    train_sequences = [
        text_to_sequence(
            text,
            word_to_index
        )
        for text in X_train
    ]

    max_length = get_max_length(train_sequences)
    print(f"Максимальная длина последовательности: {max_length}")

    y_train_encoded = encode_labels(y_train, classes)
    y_val_encoded = encode_labels(y_val, classes)

    y_test_encoded = encode_labels(y_test, classes)
    train_dataset = TextDataset(X_train, y_train_encoded, word_to_index, max_length)

    val_dataset = TextDataset(X_val, y_val_encoded, word_to_index, max_length)
    test_dataset = TextDataset(X_test, y_test_encoded, word_to_index, max_length)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = LSTMClassifier(
        vocab_size=len(word_to_index),
        embedding_dim=128,
        hidden_dim=64,
        num_classes=len(classes)
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 5
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )
        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}  "f"Train Loss: {train_loss:.4f}  "f"Val Loss: {val_loss:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.title("LSTM Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/lstm_loss.png", dpi=150)
    plt.close()

    y_true, y_pred = predict(model, test_loader, device)
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print("LSTM")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": len(word_to_index),
            "embedding_dim": 128,
            "hidden_dim": 64,
            "num_classes": len(classes),
            "max_length": max_length,
            "classes": classes
        },
        "models/lstm.pt"
    )
    with open(
        "models/tokenizer.pkl",
        "wb"
    ) as file:
        pickle.dump(
            {
                "word_to_index": word_to_index,
                "max_length": max_length
            },
            file
        )

if __name__ == "__main__":
    main()