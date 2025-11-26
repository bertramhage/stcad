import unittest
import numpy as np
from src.clustering.utils import davies_bouldin_index, euclidean_distance

class TestClusteringUtils(unittest.TestCase):

    def test_euclidean_distance(self):
        p1 = [0, 0]
        p2 = [3, 4]
        self.assertEqual(euclidean_distance(p1, p2), 5.0)
        
        p1 = np.array([0, 0])
        p2 = np.array([1, 1])
        self.assertAlmostEqual(euclidean_distance(p1, p2), np.sqrt(2))

    def test_davies_bouldin_index_simple(self):
        # Cluster 1: (0,0), (1,0) -> Centroid (0.5, 0). Radius 0.5
        # Cluster 2: (4,0), (5,0) -> Centroid (4.5, 0). Radius 0.5
        # Dist centroids: 4.0
        # Ratio: (0.5 + 0.5) / 4.0 = 0.25
        # DB Index: 0.25
        
        points = np.array([
            [0, 0], [1, 0],
            [4, 0], [5, 0]
        ])
        labels = np.array([0, 0, 1, 1])
        
        score = davies_bouldin_index(points, labels)
        self.assertAlmostEqual(score, 0.25)

    def test_davies_bouldin_index_single_cluster(self):
        points = np.array([[0, 0], [1, 1]])
        labels = np.array([0, 0])
        # Should handle gracefully, likely 0
        score = davies_bouldin_index(points, labels)
        self.assertEqual(score, 0.0)

    def test_davies_bouldin_index_perfect_separation(self):
        # If clusters are very far apart, index should be small
        points = np.array([
            [0, 0], [0.1, 0],
            [100, 0], [100.1, 0]
        ])
        labels = np.array([0, 0, 1, 1])
        
        # R1 approx 0.05, R2 approx 0.05. Dist approx 100.
        # Ratio approx 0.1 / 100 = 0.001
        score = davies_bouldin_index(points, labels)
        self.assertTrue(score < 0.01)

if __name__ == '__main__':
    unittest.main()
