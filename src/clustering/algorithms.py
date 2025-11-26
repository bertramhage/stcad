# Clustering algorithms
import numpy as np
from src.clustering.utils import euclidean_distance

class KMeans:
    def __init__(self, n_clusters: int, max_iters: int = 100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
    
    def fit(self, data: np.ndarray):
        random_indices = np.random.choice(len(data), self.n_clusters, replace=False)
        self.centroids = data[random_indices]
        
        for _ in range(self.max_iters):
            clusters = self._assign_clusters(data)
            new_centroids = self._update_centroids(data, clusters)
            
            if np.all(self.centroids == new_centroids): # Check for convergence
                break
            self.centroids = new_centroids
            
    def predict(self, data: np.ndarray):
        return self._assign_clusters(data)
            
    def _assign_clusters(self, data: np.ndarray):
        clusters = []
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
            cluster_idx = np.argmin(distances)
            clusters.append(cluster_idx)
        return np.array(clusters)
    
    def _update_centroids(self, data: np.ndarray, clusters: np.ndarray):
        new_centroids = []
        for i in range(self.n_clusters):
            cluster_points = data[clusters == i]
            if len(cluster_points) > 0:
                new_centroid = np.mean(cluster_points, axis=0)
            else:
                new_centroid = self.centroids[i] # Keep old centroid if no points assigned
            new_centroids.append(new_centroid)
        return np.array(new_centroids)

class DBSCAN:
    def __init__(self, eps: float, min_samples: int):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
    
    def fit(self, data: np.ndarray):
        n_points = len(data)
        # -2: undefined, -1: noise, >=0: cluster
        self.labels = -2 * np.ones(n_points, dtype=int)
        cluster_id = 0
        
        for point_idx in range(n_points):
            if self.labels[point_idx] != -2:
                continue # Already processed
            
            neighbors = self._region_query(data, point_idx)
            if len(neighbors) < self.min_samples:
                self.labels[point_idx] = -1 # Mark as noise
            else:
                self._expand_cluster(data, point_idx, neighbors, cluster_id)
                cluster_id += 1
                
    def _region_query(self, data: np.ndarray, point_idx: int):
        neighbors = []
        for idx in range(len(data)):
            if euclidean_distance(data[point_idx], data[idx]) <= self.eps:
                neighbors.append(idx)
        return neighbors
    
    def _expand_cluster(self, data: np.ndarray, point_idx: int, neighbors: list, cluster_id: int):
        self.labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            if self.labels[neighbor_idx] == -1:
                 # Previously noise, now border point
                 self.labels[neighbor_idx] = cluster_id
            
            elif self.labels[neighbor_idx] == -2:
                # Unvisited
                self.labels[neighbor_idx] = cluster_id
                neighbor_neighbors = self._region_query(data, neighbor_idx)
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors += neighbor_neighbors
            
            i += 1
            
