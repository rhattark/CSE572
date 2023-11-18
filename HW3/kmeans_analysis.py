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

def cosine_distance(first, second):
    return 1 - np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))

def generalized_jaccard_distance(first, second):
    min_sum = np.sum(np.minimum(first, second))
    max_sum = np.sum(np.maximum(first, second))
    
    if max_sum == 0:
        return 0  # Generalized Jaccard distance is 0 when both multisets are empty
    else:
        return 1.0 - min_sum / max_sum

def sse(X, centroids, pts_in_centroids):
    sse = 0

    for i, centroid in enumerate(centroids):
        cluster_pts = X[pts_in_centroids[i]]
        sse += np.sum((cluster_pts - centroid) ** 2)

    return sse

def kmeans_default(X, y, num_clusters, distance_fn, max_iterations):
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

        if np.all(np.isclose(centroids, new_centroids)):
            print("\nConverged at iteration", i)
            break

        centroids = new_centroids

    sse_value = sse(X, centroids, pts_in_centroids)
    print('Sum of squared error: ', sse_value)
        
    return new_centroids



if __name__ == "__main__":
    X, y = read_csv_get_X_y()
    num_clusters = get_num_clusters(y)

    print('Euclidean Distance:')
    centroids = kmeans_default(X, y, num_clusters=num_clusters, distance_fn=euclidean_distance, max_iterations=150)
    print()

    print('Cosine Distance:')
    centroids = kmeans_default(X, y, num_clusters=num_clusters, distance_fn=cosine_distance, max_iterations=150)
    print()

    print('Generalized Jaccard Distance:')
    centroids = kmeans_default(X, y, num_clusters=num_clusters, distance_fn=generalized_jaccard_distance, max_iterations=150)
    print()