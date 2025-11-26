# Various utility functions for clustering
import numpy as np

def euclidean_distance(point1, point2):
    """ Compute the Euclidean distance between two points. """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def davies_bouldin_index(points, labels):
    
    def compute_centroid(cluster_points):
        return np.mean(cluster_points, axis=0)
    
    def compute_avg_radius(cluster_points, centroid):
        return 1/len(cluster_points) \
            * np.sum(np.linalg.norm(cluster_points - centroid, axis=1)) # Average Euclidean distance
            
    unique_labels = np.unique(labels)
    centroids = []
    radii = []
    for label in unique_labels:
        cluster_points = points[labels == label]
        centroid = compute_centroid(cluster_points)
        centroids.append(centroid)
        radius = compute_avg_radius(cluster_points, centroid)
        radii.append(radius)
    
    db_index = 0.0
    for i in range(len(unique_labels)):
        max_ratio = 0.0
        for j in range(len(unique_labels)):
            if i != j:
                dist = np.linalg.norm(centroids[i] - centroids[j])
                ratio = (radii[i] + radii[j]) / dist
                if ratio > max_ratio:
                    max_ratio = ratio
        db_index += max_ratio
        
    return db_index / len(unique_labels)