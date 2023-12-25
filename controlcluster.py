from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np

# cluster coordinates using DBSCAN
def cluster_coordinates(coordinates, sigma=0.0001, min_samples=4):
    dbscan = DBSCAN(eps=sigma, min_samples=min_samples)
    clusters = list(dbscan.fit_predict(np.array(coordinates)))
    return clusters

def cluster_auto_sigma(coordinates, min_samples=4):
    array_coordinates = np.array(coordinates)

    silhouette_scores = []

    start_sigma = 0.0001
    # Calculate the silhouette score for each value of k
    for i in range(10):
        sigma = start_sigma + i / 100000
        clusters = np.array(cluster_coordinates(coordinates, sigma))

        indexes = list(np.where(clusters != -1)[0])

        # calculate silhouette score
        removed_coordinates = array_coordinates[indexes]
        removed_clusters = list(np.array(clusters)[indexes])

        try:
            score = silhouette_score(removed_coordinates, removed_clusters)
            silhouette_scores.append(score)
        except:
            print(sigma)
            print(removed_clusters)
            silhouette_scores.append(0)
            # break

    # Find the optimal number of clusters (k) with the highest silhouette score
    optimal_k = np.argmax(silhouette_scores)
    optimal_sigma = start_sigma + optimal_k / 100000

    print("Optimal number of clusters:", optimal_k, optimal_sigma)

    # cluster with optimal sigma
    return cluster_coordinates(coordinates, sigma=optimal_sigma, min_samples=min_samples)

def get_list_by_cluster(clusters):
    prev_point = -2
    temp_cluster = []
    list_by_cluster = []
    for i, c in enumerate(clusters):
        if prev_point == -2:
            temp_cluster.append(c)
            prev_point = c
        elif c == prev_point:
            temp_cluster.append(c)
            prev_point = c
        else:
            list_by_cluster.append(temp_cluster)
            prev_point = c
            temp_cluster = [c]
    list_by_cluster.append(temp_cluster)
    return list_by_cluster

def convert_clusters_to_center(coordinates, clusters):

    new_coordinates = []
    prev_point = -2
    temp_coordinates = []
    array_coordinates = np.array(coordinates)

    for i, c in enumerate(clusters):
        if prev_point == -2:
            temp_coordinates.append(coordinates[i])
            prev_point = c
        elif c == prev_point:
            temp_coordinates.append(coordinates[i])
            prev_point = c
        else:
            if prev_point == -1:
                new_coordinates += temp_coordinates
            else:
                center = np.mean(array_coordinates[clusters == prev_point], axis=0)
                new_coordinates.append(tuple(center))
            prev_point = c
            temp_coordinates = [coordinates[i]]
    if prev_point == -1:
        new_coordinates += temp_coordinates
    else:
        new_coordinates.append(tuple(np.mean(temp_coordinates, axis=0)))
    return new_coordinates


def centroid_clusters(coordinates, clusters, path):
    list_by_clusters = get_list_by_cluster(clusters)

def isroad(coordinates, threshold=3):
    # Fit PCA on the dataset
    pca = PCA(n_components=2)
    print(coordinates)
    pca.fit(coordinates)

    # Get the principal components
    # principal_components = pca.components_
    sigmas = pca.singular_values_ / np.sqrt(coordinates.shape[0])
    return sigmas[0]/sigmas[1] > threshold

def detect_roads(coordinates, clusters, threshold=3):
    array_coordinates = np.array(coordinates)
    array_clusters = np.array(clusters)
    road_ids = []
    for i in range(max(clusters) + 1):
        print(i)
        print(clusters)
        if clusters.count(i) >= 5 and isroad(array_coordinates[array_clusters == i], threshold=threshold):
            road_ids.append(i)

    new_clusters = [-1 if x in road_ids else x for x in clusters]
    return new_clusters

