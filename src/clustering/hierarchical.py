import numpy as np
from argparse import ArgumentParser
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from src.clustering.utils import davies_bouldin_index
from src.utils.logging import CustomLogger
import matplotlib.colors as mcolors
import os

class AgglomerativeClustering:
    """
    A custom implementation of Agglomerative Hierarchical Clustering.
    
    Parameters:
    -----------
    linkage : str, default='centroid'
        The linkage criterion to use.
        - 'centroid': Distance between the centroids of two clusters.
        - 'single': Minimum distance between any single point in one cluster and any point in the other.
        - 'complete': Maximum distance between any single point in one cluster and any point in the other.
        - 'average': Average distance between all points in one cluster and all points in the other.
    """
    def __init__(self, linkage='centroid'):
        self.linkage = linkage
        self.linkage_matrix_ = None # Stores the (n-1) x 4 linkage matrix for dendrograms

    def _euclidean_distance(self, a, b):
        return np.linalg.norm(a - b)

    def _compute_distance(self, cluster1_indices, cluster2_indices, X):
        points1 = X[cluster1_indices]
        points2 = X[cluster2_indices]

        if self.linkage == 'centroid':
            centroid1 = np.mean(points1, axis=0)
            centroid2 = np.mean(points2, axis=0)
            return self._euclidean_distance(centroid1, centroid2)

        elif self.linkage == 'single':
            min_dist = np.inf
            for p1 in points1:
                for p2 in points2:
                    dist = self._euclidean_distance(p1, p2)
                    if dist < min_dist:
                        min_dist = dist
            return min_dist

        elif self.linkage == 'complete':
            max_dist = -1
            for p1 in points1:
                for p2 in points2:
                    dist = self._euclidean_distance(p1, p2)
                    if dist > max_dist:
                        max_dist = dist
            return max_dist

        elif self.linkage == 'average':
            distances = []
            for p1 in points1:
                for p2 in points2:
                    distances.append(self._euclidean_distance(p1, p2))
            return np.mean(distances)
        else:
            raise ValueError(f"Unknown linkage type: {self.linkage}")

    def fit(self, X):
        """
        Fit the model and build the Linkage Matrix.
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Dictionary to store active clusters: {cluster_id: [list of sample indices]}
        # We start with clusters 0 to n-1
        current_clusters = {i: [i] for i in range(n_samples)}
        
        # List to store the history of merges in Linkage Matrix format
        # [idx1, idx2, distance, sample_count]
        self.linkage_matrix_ = []
        
        # Counter for the next cluster ID (starts after the last sample index)
        next_cluster_id = n_samples

        # Loop until one cluster remains
        while len(current_clusters) > 1:
            active_ids = list(current_clusters.keys())
            min_dist = np.inf
            best_pair = None

            # Find closest pair of active clusters
            # O(N^3) naive implementation
            for i in range(len(active_ids)):
                for j in range(i + 1, len(active_ids)):
                    id1 = active_ids[i]
                    id2 = active_ids[j]
                    
                    dist = self._compute_distance(current_clusters[id1], current_clusters[id2], X)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (id1, id2)

            # Perform the merge
            c1, c2 = best_pair
            
            # Create new cluster
            new_indices = current_clusters[c1] + current_clusters[c2]
            
            # Record merge in standard Linkage Matrix format
            # We cast to float for consistency with scipy
            self.linkage_matrix_.append([float(c1), float(c2), float(min_dist), len(new_indices)])
            
            # Update active clusters
            del current_clusters[c1]
            del current_clusters[c2]
            current_clusters[next_cluster_id] = new_indices
            
            next_cluster_id += 1

        self.linkage_matrix_ = np.array(self.linkage_matrix_)
        return self

    def get_labels(self, n_clusters):
        """
        Replays the clustering history to return labels for a specific number of clusters.
        """
        if self.linkage_matrix_ is None:
            raise RuntimeError("Model must be fit before getting labels.")
            
        n_samples = len(self.linkage_matrix_) + 1
        if n_clusters < 1 or n_clusters > n_samples:
            raise ValueError(f"n_clusters must be between 1 and {n_samples}")

        # Reconstruct state
        # 1. Start with singletons
        current_clusters = {i: [i] for i in range(n_samples)}
        
        # 2. We perform (n_samples - n_clusters) merges to reach the desired state
        num_merges = n_samples - n_clusters
        
        for i in range(num_merges):
            # linkage_matrix row: [c1, c2, dist, size]
            row = self.linkage_matrix_[i]
            c1, c2 = int(row[0]), int(row[1])
            new_id = n_samples + i
            
            # Merge
            current_clusters[new_id] = current_clusters[c1] + current_clusters[c2]
            del current_clusters[c1]
            del current_clusters[c2]

        # 3. Assign labels
        labels = np.zeros(n_samples, dtype=int)
        # Sort keys to ensure deterministic labeling order
        for label_id, cluster_id in enumerate(sorted(current_clusters.keys())):
            for sample_idx in current_clusters[cluster_id]:
                labels[sample_idx] = label_id
                
        return labels

    def plot_dendrogram(self, show_plot=False, save_path=None, p=30, color_by_size=True, **kwargs):
        """
        Plots the dendrogram with truncation and size-based coloring.
        """
        if self.linkage_matrix_ is None:
            print("Error: Model is not fit yet.")
            return

        plt.figure(figsize=(12, 6))
        plt.title(f"Hierarchical Clustering Dendrogram (Truncated to last {p} clusters)")
        plt.xlabel("Cluster Size / Sample Index")
        plt.ylabel("Distance")

        # --- Feature 1: Coloring by Cluster Size ---
        link_color_func = None
        if color_by_size:
            n_samples = len(self.linkage_matrix_) + 1
            cluster_sizes = {i: 1 for i in range(n_samples)}
            
            for i, row in enumerate(self.linkage_matrix_):
                cluster_idx = n_samples + i
                cluster_sizes[cluster_idx] = int(row[3])

            cmap = plt.get_cmap('viridis') 
            norm = mcolors.LogNorm(vmin=1, vmax=n_samples)

            def size_color_func(k):
                size = cluster_sizes.get(k, 1)
                return mcolors.to_hex(cmap(norm(size)))
                
            link_color_func = size_color_func

        # --- Feature 2: Truncation ---
        dendrogram(
            self.linkage_matrix_,
            truncate_mode='lastp',
            p=p,
            show_contracted=True,
            leaf_rotation=90.,
            leaf_font_size=10.,
            show_leaf_counts=True,
            link_color_func=link_color_func,
            **kwargs
        )
        
        # --- FIXED SECTION BELOW ---
        if color_by_size:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            # Get the current Axes explicitly
            ax = plt.gca()
            
            # Tell colorbar which axes to steal space from (ax=ax)
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Number of Samples in Cluster (Log Scale)')

        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        if show_plot:
            plt.show()

class CURE:
    """
    CURE (Clustering Using REpresentatives) Implementation.
    
    Scales Agglomerative Clustering to large datasets by:
    1. Clustering a random sample of the data.
    2. Identifying representative points for each cluster.
    3. Assigning the remaining data to the closest representative.
    """
    def __init__(self, sample_size=200, n_representatives=4, compression=0.5, linkage='centroid'):
        self.sample_size = sample_size
        self.n_representatives = n_representatives
        self.compression = compression # Alpha: how much to shrink towards centroid
        self.linkage = linkage
        self.agg_ = None
        self.X_sample_ = None
        self.X_ = None # Store reference to full dataset

    def _euclidean_distance(self, a, b):
        return np.linalg.norm(a - b)

    def fit(self, X):
        """
        Samples the data and builds the hierarchical cluster tree on the sample.
        Does NOT assign labels to the full dataset yet. Use get_labels(k) for that.
        """
        self.X_ = np.array(X)
        n_samples = self.X_.shape[0]

        # 1. Sampling Step
        # If data is smaller than sample size, just use all data
        if n_samples <= self.sample_size:
            sample_indices = np.arange(n_samples)
            self.X_sample_ = self.X_
        else:
            # Randomly select sample points
            sample_indices = np.random.choice(n_samples, self.sample_size, replace=False)
            self.X_sample_ = self.X_[sample_indices]

        # 2. Cluster the Sample (The expensive part, but on small N)
        # We reuse our custom AgglomerativeClustering class which builds the full tree
        self.agg_ = AgglomerativeClustering(linkage=self.linkage)
        self.agg_.fit(self.X_sample_)
        
        return self

    def get_labels(self, n_clusters):
        """
        Generates labels for the full dataset by cutting the sample tree at n_clusters,
        calculating representatives, and assigning all points.
        """
        if self.agg_ is None:
            raise RuntimeError("Run fit() before get_labels()")

        # 1. Get sample labels from the hierarchical tree (fast tree cut)
        sample_labels = self.agg_.get_labels(n_clusters)

        # 2. Select Representatives (Recalculated for this specific k)
        representatives = []
        
        for k in range(n_clusters):
            # Get points in this cluster
            cluster_points = self.X_sample_[sample_labels == k]
            
            if len(cluster_points) == 0:
                continue

            # Calculate Centroid
            centroid = np.mean(cluster_points, axis=0)

            # Select Representatives
            reps = []
            if len(cluster_points) <= self.n_representatives:
                 reps = [p for p in cluster_points]
            else:
                # Standard CURE selection:
                # 1. First point is farthest from centroid
                dists = [self._euclidean_distance(p, centroid) for p in cluster_points]
                first_idx = np.argmax(dists)
                reps.append(cluster_points[first_idx])
                
                # 2. Subsequent points farthest from existing reps
                for _ in range(self.n_representatives - 1):
                    max_min_dist = -1
                    best_candidate = None
                    
                    for p in cluster_points:
                        # Find min distance to any existing rep
                        min_dist_to_reps = min([self._euclidean_distance(p, r) for r in reps])
                        
                        if min_dist_to_reps > max_min_dist:
                            max_min_dist = min_dist_to_reps
                            best_candidate = p
                    
                    if best_candidate is not None:
                        reps.append(best_candidate)

            # Shrink representatives towards centroid
            shrunk_reps = []
            for r in reps:
                # Move r towards centroid by compression factor
                new_pos = r + self.compression * (centroid - r)
                shrunk_reps.append(new_pos)
            
            representatives.append(shrunk_reps)

        # 3. Assign entire dataset (Linear Scan)
        n_samples = self.X_.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        # This linear scan is O(N * k * n_reps), which is much faster than clustering
        for i in range(n_samples):
            point = self.X_[i]
            min_dist = np.inf
            best_cluster = -1
            
            # Check distance to ALL representatives of ALL clusters
            for cluster_idx, reps in enumerate(representatives):
                for r in reps:
                    dist = self._euclidean_distance(point, r)
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = cluster_idx
            
            labels[i] = best_cluster
            
        return labels
    
    def plot_dendrogram(self, show_plot=False, save_path: str | None = None, **kwargs):
        """
        Plots the dendrogram of the sample clustering.
        
        Parameters:
        -----------
        show_plot : bool, default=False
            Whether to display the plot.
        save_path : str | None, default=None
            If provided, saves the plot to this path.
        **kwargs : 
            Arguments passed to scipy.cluster.hierarchy.dendrogram
        """
        if self.agg_ is None:
            print("Error: Model is not fit yet.")
            return
        
        self.agg_.plot_dendrogram(show_plot=show_plot, save_path=save_path, **kwargs)
    
if __name__ == "__main__":
    parser = ArgumentParser("CURE Hierarchical Clustering for Large Datasets. Build and save entire tree.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset (embeddings).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the linkage matrix (npz file).")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of points to sample for clustering.")
    parser.add_argument("--n_representatives", type=int, default=20, help="Number of representatives per cluster.")
    parser.add_argument("--compression", type=float, default=0.2, help="Compression factor towards centroid (alpha).")
    parser.add_argument("--linkage", type=str, default="single", choices=["centroid", "single", "complete", "average"], help="Linkage criterion for hierarchical clustering.")
    parser.add_argument("--dendrogram_p", type=int, default=30, help="Number of last merges to show in dendrogram plot.")
    args = parser.parse_args()
    
    logger = CustomLogger(project_name='Computational-Tools', group='clustering', run_name='run_cure', use_wandb=True)
    
    logger.log_config(vars(args))
    
    logger.info("Loading embeddings...")
    data = np.load(args.data_path)
    embeddings = data["embeddings"]
    del data
    logger.info(f"Loaded {embeddings.shape[0]} samples with dimension {embeddings.shape[1]} from {args.data_path}\n")
    
    logger.info("Fitting CURE hierarchical clustering...")
    cure = CURE(sample_size=args.sample_size, n_representatives=args.n_representatives, compression=args.compression, linkage=args.linkage)
    cure.fit(embeddings)
    logger.info("Fitted CURE model.\n")
    
    logger.info(f"Saving linkage matrix and sample data to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    model_path = os.path.join(args.output_path, "cure_model.npz")
    np.savez_compressed(
        args.output_path, 
        linkage_matrix=cure.agg_.linkage_matrix_,
        sample_data=cure.X_sample_
    )
    
    logger.info("Saving dendrogram plot...")
    dendrogram_path = os.path.join(args.output_path, "dendrogram.png")
    cure.plot_dendrogram(save_path=dendrogram_path, p=args.dendrogram_p)
    
    logger.artifact("dendrogram_plot", dendrogram_path, "image")
    
    scores = []
    for n_clusters in [2, 5, 10, 20]:
                    labels = cure.get_labels(n_clusters=n_clusters)
                    score = davies_bouldin_index(embeddings, labels)
                    scores.append(score)
    
    logger.info("Davies-Bouldin Index scores for different cluster counts:")
    for n_clusters, score in zip([2, 5, 10, 20], scores):
        logger.info(f" - {n_clusters} clusters: DBI = {score:.4f}")
        
    logger.log_summary({"davies_bouldin_index": {n: s for n, s in zip([2, 5, 10, 20], scores)}})
    
    logger.info("Done.")