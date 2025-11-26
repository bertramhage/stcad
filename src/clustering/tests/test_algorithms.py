import unittest
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import adjusted_rand_score
from src.clustering.algorithms import KMeans, DBSCAN

class TestAlgorithms(unittest.TestCase):

    def test_kmeans_vs_sklearn(self):
        # Generate synthetic data
        X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
        
        # Custom KMeans
        kmeans = KMeans(n_clusters=3, max_iters=100)
        kmeans.fit(X)
        custom_labels = kmeans.predict(X)
        
        # Sklearn KMeans
        sklearn_kmeans = SklearnKMeans(n_clusters=3, random_state=42, n_init=10)
        sklearn_kmeans.fit(X)
        sklearn_labels = sklearn_kmeans.labels_
        
        # Compare using Adjusted Rand Index
        ari = adjusted_rand_score(sklearn_labels, custom_labels)
        print(f"KMeans ARI: {ari}")
        self.assertGreater(ari, 0.8, "KMeans result should be similar to sklearn")

    def test_dbscan_vs_sklearn(self):
        # Generate synthetic data (moons are good for DBSCAN)
        X, y = make_moons(n_samples=200, noise=0.05, random_state=42)
        
        eps = 0.3
        min_samples = 5
        
        # Custom DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X)
        custom_labels = dbscan.labels
        
        # Sklearn DBSCAN
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_samples)
        sklearn_dbscan.fit(X)
        sklearn_labels = sklearn_dbscan.labels_
        
        # Compare labels
        ari = adjusted_rand_score(sklearn_labels, custom_labels)
        print(f"DBSCAN ARI: {ari}")
        self.assertGreater(ari, 0.9, "DBSCAN result should be similar to sklearn")

if __name__ == '__main__':
    unittest.main()
