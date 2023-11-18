import pandas as pd

def read_csv_get_X_y():
    X = pd.read_csv('kmeans_data/data.csv')
    y = pd.read_csv('kmeans_data/label.csv')
    return X, y

def kmeans(X, y):
    pass

if __name__ == "__main__":
    X, y = read_csv_get_X_y()
    kmeans(X, y)