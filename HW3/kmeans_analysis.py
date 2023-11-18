import numpy as np

def read_csv_get_X_y():
    X = np.genfromtxt('kmeans_data/data.csv', delimiter=',')
    y = np.genfromtxt('kmeans_data/label.csv', delimiter=',')
    return X, y

def get_num_clusters(y):
    return len(np.unique(y))

def kmeans(X, y, num_clusters):
    X_len = len(X)


if __name__ == "__main__":
    X, y = read_csv_get_X_y()
    num_clusters = get_num_clusters(y)
    kmeans(X, y, num_clusters)