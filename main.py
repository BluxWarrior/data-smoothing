import controlcluster
import controlpath
import numpy as np
import matplotlib.pyplot as plt

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

file_path = "./GPS_DATA/walks/walkitlikedog.json"

original_coordinates = controlpath.extract_coordinates(file_path)
# print(controlpath.encode_coordinates(original_coordinates))
original_pathID = np.arange(len(original_coordinates))
original_path = controlpath.extract_path(original_coordinates, original_pathID)

clusters = controlcluster.cluster_auto_sigma(original_coordinates, min_samples=2)
# print(clusters)
detected_clusters = controlcluster.detect_roads(original_coordinates, clusters, threshold=3)
new_coordinates = controlcluster.convert_clusters_to_center(original_coordinates, detected_clusters)

plot_coordinates(original_coordinates, detected_clusters)
plot_coordinates(new_coordinates, view_path=True)


plt.show()