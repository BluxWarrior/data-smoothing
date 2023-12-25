import re
import json
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def extract_coordinates(coord_string):
    match = re.search(r'POINT\(([^ ]+) ([^ ]+)\)', coord_string)
    if match:
        return float(match.group(1)), float(match.group(2))
    else:
        return None, None

def encode_coordinates(coords):
    def encode_value(value):
        value = int(round(value * 1e5))
        value <<= 1
        if value < 0:
            value = ~value
        chunks = []
        while value >= 0x20:
            chunks.append((0x20 | (value & 0x1f)) + 63)
            value >>= 5
        chunks.append(value + 63)
        return chunks

    encoded_string = ""
    last_lat, last_lng = 0, 0
    for lat, lng in coords:
        delta_lat = encode_value(lat - last_lat)
        delta_lng = encode_value(lng - last_lng)
        last_lat, last_lng = lat, lng
        encoded_string += ''.join(map(chr, delta_lat + delta_lng))
    return encoded_string

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
            print(i)
            print(removed_clusters)
            silhouette_scores.append(0)
            # break

    # Find the optimal number of clusters (k) with the highest silhouette score
    optimal_k = np.argmax(silhouette_scores)
    optimal_sigma = start_sigma + optimal_k / 100000

    print("Optimal number of clusters:", optimal_k, optimal_sigma)

    # cluster with optimal sigma
    return cluster_coordinates(original_coordinates, optimal_sigma, min_samples)
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

def filter_clusters(clusters):
    # get the list that is seperated by clusters
    list_by_cluster = get_list_by_cluster(clusters)
    print(list_by_cluster)

    current_cluster = -1

    new_clusters = []
    for cluster in list_by_cluster:
        if cluster[0] == -1:
            new_clusters += cluster
        elif len(cluster) < 3:
            new_clusters += [-1 for _ in cluster]
        elif cluster[0] > current_cluster:
            new_clusters += cluster
            current_cluster = cluster[0]
        elif cluster[0] <= current_cluster:
            current_cluster += 1
            new_clusters += [current_cluster for _ in cluster]
    return new_clusters

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

def plot_coordinates(coordinates, clusters=None, view_path=False, view_endpoints = True):
    array_coordinates = np.array(coordinates)
    array_clusters = np.array(clusters)

    # Visualization with matplotlib
    plt.figure(figsize=(12, 6))
    # Plot original path data
    plt.scatter(array_coordinates[:, 0], array_coordinates[:, 1], c='gray', label='Path Data', alpha=0.5)

    # Plot detected clusters
    if clusters is not None:
        for cluster_id in set(clusters):
            if cluster_id > -1:
                cluster_points = array_coordinates[array_clusters == cluster_id]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, label=f'Cluster {cluster_id}')



    # Plot the paths
    if view_path:
         plt.plot(array_coordinates[:, 0], array_coordinates[:, 1], c='blue', label='Smoothed Path')

    if view_endpoints:
        start_point = array_coordinates[0]
        end_point = array_coordinates[-1]
        plt.scatter([start_point[0]], [start_point[1]], c='red', marker='o', edgecolor='black', linewidth=2, s=100,
                    label='Start Point')
        plt.scatter([end_point[0]], [end_point[1]], c='green', marker='X', edgecolor='black', linewidth=2, s=100,
                    label='End Point')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)

    plt.show(block=False)

def isroad(coordinates, threshold=3):
    # Fit PCA on the dataset
    pca = PCA(n_components=2)
    pca.fit(coordinates)

    # Get the principal components
    # principal_components = pca.components_
    sigmas = pca.singular_values_ / np.sqrt(coordinates.shape[0])
    return sigmas[0]/sigmas[1] > threshold

def detect_roads(coordinates, clusters):
    array_coordinates = np.array(coordinates)
    array_cluster = np.array(clusters)
    road_ids = []
    for i in range(max(original_clusters) + 1):
        if original_clusters.count(i) >= 5 and isroad(array_coordinates[array_cluster == i]):
            road_ids.append(i)

    new_clusters = [-1 if x in road_ids else x for x in clusters]
    return new_clusters



# def plot_data(coordinates, clusters)
# Read and parse the JSON file
file_path = './GPS_DATA/walks/57af1f74-3bbb-5298-a3a9-136b9fc94ab9.json'
with open(file_path, 'r') as file:
    data_json = json.loads(file.read())



# Extract coordinates
original_coordinates = [extract_coordinates(entry['coordinates']) for entry in data_json]

original_clusters = cluster_auto_sigma(original_coordinates, min_samples=2)

new_clusters = detect_roads(original_coordinates, original_clusters)

new_coordinates = convert_clusters_to_center(original_coordinates, new_clusters)

longitudes, latitudes = zip(*new_coordinates)
print(new_coordinates)
print(encode_coordinates(zip(latitudes, longitudes)))
plot_coordinates(original_coordinates)
plot_coordinates(new_coordinates, view_path=True)

plt.show()
