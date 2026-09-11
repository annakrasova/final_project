import pandas as pd
from sklearn.model_selection import train_test_split
DATA_PATH = "data/articles.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[
        ["prod_name", "detail_desc", "garment_group_name"]
    ]
    print(df.shape)
    return df

def edit_data(df):
    df = df.dropna(subset=["prod_name", "detail_desc", "garment_group_name"])
    top_5_classes = (df["garment_group_name"].value_counts().head(5).index.tolist())
    df = df[df["garment_group_name"].isin(top_5_classes)]
    max_size = 10000
    if len(df) > max_size:
        df = df.sample(n=max_size, random_state=42)
    return df, top_5_classes

def create_text(df):
    df = df.copy()
    df["text"] = (df["prod_name"] + " " + df["detail_desc"])
    return df

def x_and_y(df):
    X = df["text"]
    y = df["garment_group_name"]
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def data_func():
    df = load_data()
    df, top_classes = edit_data(df)
    df = create_text(df)
    X, y = x_and_y(df)
    print(top_classes)
    return top_classes, X, y, df

if __name__ == "__main__":
    classes, X, y, df = data_func()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("\nРазмеры выборок:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    pd.set_option('display.max_colwidth', None)
    print(df.head(5))