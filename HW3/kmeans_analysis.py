import numpy as np
from collections import Counter

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
    np.random.seed(69)
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
        
    return new_centroids, pts_in_centroids

def kmeans_until_convergence(X, y, num_clusters, distance_fn):
    np.random.seed(69)
    X_len = len(X)
    random_idx_list = np.random.choice(range(X_len), size=num_clusters, replace=False)
    centroids = X[random_idx_list]
    iter = 0

    while True:
        print(f"Iteration {iter}", end='\r')
        iter += 1
        pts_in_centroids = [[] for _ in range(num_clusters)]

        for pt_idx in range(X_len):
            distances = [distance_fn(centroid, X[pt_idx]) for centroid in centroids]
            idx_closest_centroid = np.argmin(distances)
            pts_in_centroids[idx_closest_centroid].append(pt_idx)

        new_centroids = [X[cent_group].mean(axis=0) for cent_group in pts_in_centroids]

        if np.all(np.isclose(centroids, new_centroids)):
            print("\nConverged at iteration", iter)
            break

        centroids = new_centroids

    sse_value = sse(X, centroids, pts_in_centroids)
        
    return new_centroids, pts_in_centroids, sse_value

def kmeans_until_sse_inc(X, y, num_clusters, distance_fn):
    np.random.seed(69)
    X_len = len(X)
    random_idx_list = np.random.choice(range(X_len), size=num_clusters, replace=False)
    centroids = X[random_idx_list]
    iter = 0
    cur_sse = 694206942069420

    while True:
        print(f"Iteration {iter}", end='\r')
        iter += 1
        pts_in_centroids = [[] for _ in range(num_clusters)]

        for pt_idx in range(X_len):
            distances = [distance_fn(centroid, X[pt_idx]) for centroid in centroids]
            idx_closest_centroid = np.argmin(distances)
            pts_in_centroids[idx_closest_centroid].append(pt_idx)

        new_centroids = [X[cent_group].mean(axis=0) for cent_group in pts_in_centroids]
        new_sse = sse(X, centroids, pts_in_centroids)

        # additional condition will mark both sse similar if they're same to the 4th decimal position
        # to avoid infinite looping
        if new_sse > cur_sse or round(new_sse, 4) == round(cur_sse, 4):
            print("\nConverged at iteration", iter)
            break

        centroids = new_centroids
        cur_sse = new_sse

    sse_value = sse(X, centroids, pts_in_centroids)
        
    return new_centroids, pts_in_centroids, sse_value

def get_labels_clustered_data(X, y, pts_in_centroids):
    labeled_data = []
    
    for group in pts_in_centroids:
        counter = Counter(y[group])
        label = counter.most_common(1)[0][0]
        group_labeled_data = [[idx, label] for idx in group]
        labeled_data.extend(group_labeled_data)

    labeled_data.sort(key=lambda x : x[0])

    return np.array([label for idx, label in labeled_data])

def get_accuracy(first, second):
    correct_predictions = np.sum(first == second)
    size = len(first)
    return correct_predictions / size * 100

if __name__ == "__main__":
    X, y = read_csv_get_X_y()
    num_clusters = get_num_clusters(y)

    # Q1

    # print('Euclidean Distance:')
    # euc_centroids, euc_pts_in_centroids = kmeans_default(X, y, num_clusters=num_clusters, distance_fn=euclidean_distance, max_iterations=150)
    # print()

    # print('Cosine Distance:')
    # cos_centroids, cos_pts_in_centroids = kmeans_default(X, y, num_clusters=num_clusters, distance_fn=cosine_distance, max_iterations=150)
    # print()

    # print('Generalized Jaccard Distance:')
    # jac_centroids, jac_pts_in_centroids = kmeans_default(X, y, num_clusters=num_clusters, distance_fn=generalized_jaccard_distance, max_iterations=150)
    # print()

    # Q2

    # euc_labels = get_labels_clustered_data(X, y, euc_pts_in_centroids)
    # euc_accuracy = get_accuracy(euc_labels, y)
    # print(f'Accuracy with Euclidean Distance: {euc_accuracy}%')

    # cos_labels = get_labels_clustered_data(X, y, cos_pts_in_centroids)
    # cos_accuracy = get_accuracy(cos_labels, y)
    # print(f'Accuracy with Cosine Distance: {cos_accuracy}%')

    # jac_labels = get_labels_clustered_data(X, y, jac_pts_in_centroids)
    # jac_accuracy = get_accuracy(jac_labels, y)
    # print(f'Accuracy with Generalized Jaccard Distance: {jac_accuracy}%')

    # Q3

    # print('Euclidean Distance:')
    # euc_conv_centroids, euc_conv_pts_in_centroids, euc_conv_sse = kmeans_until_convergence(X, y, num_clusters=num_clusters, distance_fn=euclidean_distance)
    # print()

    # print('Cosine Distance:')
    # cos_conv_centroids, cos_conv_pts_in_centroids, cos_conv_sse = kmeans_until_convergence(X, y, num_clusters=num_clusters, distance_fn=cosine_distance)
    # print()

    # print('Generalized Jaccard Distance:')
    # jac_conv_centroids, jac_conv_pts_in_centroids, jac_conv_sse = kmeans_until_convergence(X, y, num_clusters=num_clusters, distance_fn=generalized_jaccard_distance)
    # print()

    print('Euclidean Distance:')
    euc_sse_centroids, euc_sse_pts_in_centroids, euc_sse_sse = kmeans_until_sse_inc(X, y, num_clusters=num_clusters, distance_fn=euclidean_distance)
    print()

    print('Cosine Distance:')
    cos_sse_centroids, cos_sse_pts_in_centroids, cos_sse_sse = kmeans_until_sse_inc(X, y, num_clusters=num_clusters, distance_fn=cosine_distance)
    print()

    print('Generalized Jaccard Distance:')
    jac_sse_centroids, jac_sse_pts_in_centroids, jac_sse_sse = kmeans_until_sse_inc(X, y, num_clusters=num_clusters, distance_fn=generalized_jaccard_distance)
    print()

    # Q4

    # print(f'SSE - Euclidean Distance   - Until No change in centroid position: {euc_conv_sse}')
    # print(f'SSE - Cosine Distance      - Until No change in centroid position: {cos_conv_sse}')
    # print(f'SSE - Gen Jaccard Distance - Until No change in centroid position: {jac_conv_sse}')
    # print()

    print(f'SSE - Euclidean Distance   - Until SSE increases: {euc_sse_sse}')
    print(f'SSE - Cosine Distance      - Until SSE increases: {cos_sse_sse}')
    print(f'SSE - Gen Jaccard Distance - Until SSE increases: {jac_sse_sse}')
    print()