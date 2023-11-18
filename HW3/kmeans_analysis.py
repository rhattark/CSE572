import numpy as np

def read_csv_get_X_y():
    X = np.genfromtxt('kmeans_data/data.csv', delimiter=',')
    y = np.genfromtxt('kmeans_data/label.csv', delimiter=',')
    return X, y

def get_num_clusters(y):
    return len(np.unique(y))

def euclidean_distance(first, second):
    diff = first - second
    squared = diff * diff
    sum_of_squares = squared.sum()
    distance = np.sqrt(sum_of_squares)
    return distance

def kmeans_max_iterations(X, y, num_clusters, distance_fn, max_iterations):
    X_len = len(X)
    random_idx_list = np.random.choice(range(X_len), size=num_clusters, replace=False)
    centroids = X[random_idx_list]

    for i in range(max_iterations):
        print(f"Iteration {i}", end='\r')
        pts_in_centroids = [[] for _ in range(num_clusters)]

        for pt_idx in range(X_len):
            distances = [distance_fn(centroid, X[pt_idx]) for centroid in centroids]
            idx_closest_centroid = np.argmin(distances)
            pts_in_centroids[idx_closest_centroid].append(pt_idx)

        new_centroids = [X[cent_group].mean(axis=0) for cent_group in pts_in_centroids]
        
    return new_centroids



if __name__ == "__main__":
    X, y = read_csv_get_X_y()
    num_clusters = get_num_clusters(y)
    centroids = kmeans_max_iterations(X, y, num_clusters=num_clusters, distance_fn=euclidean_distance, max_iterations=50)
    print(centroids)