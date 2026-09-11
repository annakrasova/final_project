import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from data import data_func
import os
REPORTS_PATH = "reports"
os.makedirs(REPORTS_PATH, exist_ok=True)

def analyze_dataset(X,y):
    print(f"Количество объектов: {len(X)}")
    print(f"Количество классов: {y.nunique()}")
    print(f"Количество признаков: 1 (текстовый признак)")
    
def analyze_classes(y):
    print("РАСПРЕДЕЛЕНИЕ КЛАССОВ")
    class_counts = y.value_counts()
    print(class_counts)
    return class_counts

def analyze_missing_values(X, y):
    print("ПРОПУСКИ")
    print(f"Пропуски в X: {X.isna().sum()}")
    print(f"Пропуски в y: {y.isna().sum()}")

def analyze_textlength_byclasses(df):
    df = df.copy()
    df["text_length"] = df["text"].str.len()
    mean_lengths = (df.groupby("garment_group_name")["text_length"].mean().sort_values(ascending=False))
    return mean_lengths

def plot_classes(y):
    class_counts = y.value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title("Распределение товаров по классам")
    plt.xlabel("Группа товаров")
    plt.ylabel("Количество товаров")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

def plot_textlength(X):
    text_lengths = X.str.len()
    plt.figure(figsize=(10, 6))
    sns.histplot(text_lengths, bins=50)
    plt.title("Распределение длины описаний товаров")
    plt.xlabel("Количество символов")
    plt.ylabel("Количество товаров")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_PATH}/text_length_distribution.png", dpi=150)
    plt.show()

def plot_mean_textlength(mean_lengths):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=mean_lengths.index,y=mean_lengths.values)
    plt.title("Средняя длина текста по классам")
    plt.xlabel("Группа товаров")
    plt.ylabel("Средняя длина текста, символы")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_PATH}/mean_text_length_byclasses.png", dpi=150)
    plt.show()

def main():
    classes, X, y, df = data_func()
    analyze_dataset(X, y)
    analyze_classes(y)
    analyze_missing_values(X, y)
    analyze_textlength_byclasses(df)
    plot_classes(y)
    plot_textlength(X)
    mean_lengths = analyze_textlength_byclasses(df)
    plot_mean_textlength(mean_lengths)
if __name__ == "__main__":
    main()