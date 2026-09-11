from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from data import data_func
from data import split_data

def main():
    classes, X, y, df = data_func()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print("DummyClassifier")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

if __name__ == "__main__":
    main()