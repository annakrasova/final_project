import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix)
from sklearn.pipeline import Pipeline
from data import data_func, split_data

def main():
    classes, X, y, df = data_func()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = Pipeline([("tfidf",TfidfVectorizer(
                lowercase=True,
                max_features=20000,
                ngram_range=(1, 2),
                min_df=2
            )),("classifier",
            LogisticRegression(max_iter=1000))])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    joblib.dump(model, "models/logistic_pipeline.pkl" )

if __name__ == "__main__":
    main()