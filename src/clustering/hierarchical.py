import numpy as np
from argparse import ArgumentParser
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from src.clustering.utils import davies_bouldin_index
from src.utils.logging import CustomLogger
import matplotlib.colors as mcolors
from src.clustering.interactive_plot import plot_interactive_dendrogram
from src.clustering.utils import euclidean_distance, cluster_distance
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
    def __init__(self, linkage='centroid', pruning_fraction=0.1, pruning_min_size=1):
        self.linkage = linkage
        self.linkage_matrix_ = None
        self.pruning_fraction = pruning_fraction
        self.pruning_min_size = pruning_min_size
        self.noise_points_ = []

    def fit(self, X):
        """ Fit the model and build the Linkage Matrix. """
        X = np.array(X)
        n_samples = X.shape[0]
        self.n_samples_ = n_samples
        
        # Initialize clusters
        current_clusters = {i: [i] for i in range(n_samples)}
        
        self.linkage_matrix_ = []
        
        # Create a id new clusters
        next_cluster_id = n_samples
        
        pruning_triggered = False

        # Loop until one cluster remains
        while len(current_clusters) > 1:
            if not pruning_triggered and len(current_clusters) <= n_samples * self.pruning_fraction:
                pruning_triggered = True
                
                ids_to_prune = []
                for cid, indices in current_clusters.items():
                    if len(indices) <= self.pruning_min_size:
                        ids_to_prune.append(cid)
                        self.noise_points_.extend(indices)
                
                for cid in ids_to_prune:
                    del current_clusters[cid]
            
            active_ids = list(current_clusters.keys())
            min_dist = np.inf
            best_pair = None

            # Find closest pair of active clusters
            for i in range(len(active_ids)):
                for j in range(i + 1, len(active_ids)):
                    id1 = active_ids[i]
                    id2 = active_ids[j]
                    
                    dist = cluster_distance(X, current_clusters[id1], current_clusters[id2], linkage=self.linkage)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (id1, id2)
                        
            c1, c2 = best_pair
            
            # Create new cluster
            new_indices = current_clusters[c1] + current_clusters[c2]
            
            # Store [c1, c2, distance, size]
            self.linkage_matrix_.append([float(c1), float(c2), float(min_dist), len(new_indices)])
            
            # Update clusters
            del current_clusters[c1]
            del current_clusters[c2]
            current_clusters[next_cluster_id] = new_indices
            
            next_cluster_id += 1

        self.linkage_matrix_ = np.array(self.linkage_matrix_)
        return self

    def get_labels(self, n_clusters):
        """ Replays the clustering history to return labels for a specific number of clusters. """
        if self.linkage_matrix_ is None:
            raise RuntimeError("Model must be fit before getting labels.")
            
        n_samples = self.n_samples_ - len(self.noise_points_)

        # Reconstruct state at desired number of clusters
        current_clusters = {i: [i] for i in range(self.n_samples_)}
        
        num_merges = n_samples - n_clusters
        for i in range(num_merges):
            row = self.linkage_matrix_[i]
            c1, c2 = int(row[0]), int(row[1])
            new_id = self.n_samples_ + i
            
            current_clusters[new_id] = current_clusters[c1] + current_clusters[c2]
            del current_clusters[c1]
            del current_clusters[c2]

        # Assign labels
        noise_set = set(self.noise_points_)
        labels = np.full(self.n_samples_, -1, dtype=int) # Initialize as noise
        valid_label_id = 0
        for cluster_id in sorted(current_clusters.keys()):
            indices = current_clusters[cluster_id]
            if indices[0] in noise_set:
                continue
            for sample_idx in indices:
                labels[sample_idx] = valid_label_id
            valid_label_id += 1
                
        return labels

    def plot_dendrogram(self, show_plot=False, save_path=None, p=30, **kwargs):
        """ Plots the dendrogram with truncation and size-based coloring. """
        if self.linkage_matrix_ is None:
            raise RuntimeError("Model must be fit before plotting dendrogram.")

        plt.figure(figsize=(12, 6))
        plt.title(f"Hierarchical Clustering Dendrogram (Truncated to last {p} clusters)")
        plt.xlabel("Cluster Size / Sample Index")
        plt.ylabel("Distance")
        
        # Add noise points
        # 1. Identify Survivors
        # We need the set of noise indices. 
        # Ensure you populated self.noise_points_ in fit() as discussed previously!
        all_indices = set(range(self.n_samples_))
        noise_set = set(self.noise_points_) if hasattr(self, 'noise_points_') else set()
        
        # Sort survivors to maintain relative order
        survivors = sorted(list(all_indices - noise_set))
        n_survivors = len(survivors)
        
        if n_survivors < 2:
            print("Not enough points survived pruning to plot a dendrogram.")
            return

        # 2. Create the Mapping
        # Map: Old_Leaf_ID -> New_Leaf_ID (0 to M-1)
        id_map = {old_idx: new_idx for new_idx, old_idx in enumerate(survivors)}

        # 3. Translate the Linkage Matrix
        # We need to map the inputs of every row in the matrix.
        clean_Z = []
        
        # The 'fit' method generates cluster IDs starting at n_samples.
        # We need to map those Old_Cluster_IDs to New_Cluster_IDs starting at n_survivors.
        
        # Iterating through the existing valid matrix
        for i, row in enumerate(self.linkage_matrix_):
            old_c1, old_c2, dist, size = row
            old_c1, old_c2 = int(old_c1), int(old_c2)
            
            # Map the inputs
            # If input is not in map, it means we have a logic error or it's a pruned point 
            # (but pruned points shouldn't be in linkage_matrix_ in your code)
            if old_c1 not in id_map or old_c2 not in id_map:
                continue # Skip rows that somehow reference noise (safety check)

            new_c1 = id_map[old_c1]
            new_c2 = id_map[old_c2]
            
            clean_Z.append([float(new_c1), float(new_c2), dist, size])
            
            # The result of this row creates a new cluster.
            # Old ID: self.n_samples_ + i
            # New ID: n_survivors + len(clean_Z) - 1
            old_result_id = self.n_samples_ + i
            new_result_id = n_survivors + len(clean_Z) - 1
            
            # Update map for future rows that might reference this cluster
            id_map[old_result_id] = new_result_id

        clean_Z = np.array(clean_Z)
        # Coloring by Cluster Size
        link_color_func = None
        n_samples = len(clean_Z) + 1
        cluster_sizes = {i: 1 for i in range(n_samples)}
        
        for i, row in enumerate(clean_Z):
            cluster_idx = n_samples + i
            cluster_sizes[cluster_idx] = int(row[3])

        cmap = plt.get_cmap('viridis') 
        norm = mcolors.LogNorm(vmin=1, vmax=n_samples)

        def size_color_func(k):
            size = cluster_sizes.get(k, 1)
            return mcolors.to_hex(cmap(norm(size)))
            
        link_color_func = size_color_func

        dendrogram(
            clean_Z,
            truncate_mode='lastp',
            p=p,
            show_contracted=True,
            leaf_rotation=90.,
            leaf_font_size=10.,
            show_leaf_counts=True,
            link_color_func=link_color_func,
            **kwargs
        )

        # Create colorbar for cluster sizes
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        ax = plt.gca()
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
    def __init__(self, sample_size=200,
                 n_representatives=4,
                 compression=0.5,
                 linkage='centroid',
                 pruning_fraction=0.1,
                 pruning_min_size=1,
                 assignment_threshold=None,
                 seed=42):
        self.sample_size = sample_size
        self.n_representatives = n_representatives
        self.compression = compression # Alpha: how much to shrink towards centroid
        self.linkage = linkage
        self.pruning_fraction = pruning_fraction
        self.pruning_min_size = pruning_min_size
        self.assignment_threshold = assignment_threshold if assignment_threshold is not None else np.inf
        self.seed = seed
        self.agg_ = None
        self.X_sample_ = None
        self.X_ = None # Store reference to full dataset

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
            np.random.seed(self.seed)
            sample_indices = np.random.choice(n_samples, self.sample_size, replace=False)
            self.X_sample_ = self.X_[sample_indices]

        # 2. Cluster the Sample (The expensive part, but on small N)
        # We reuse our custom AgglomerativeClustering class which builds the full tree
        self.agg_ = AgglomerativeClustering(linkage=self.linkage, pruning_fraction=self.pruning_fraction, pruning_min_size=self.pruning_min_size)
        self.agg_.fit(self.X_sample_)
        
        return self
    
    def save(self, save_path: str):
        """
        Saves the CURE model to an .npz file.
        
        Parameters:
        -----------
        save_path : str
            Path to save the .npz file containing 'linkage_matrix' and 'sample_data'.
        """
        if self.agg_ is None or self.X_sample_ is None:
            raise RuntimeError("Model must be fit before saving.")
        
        meta_data = {
            "n_representatives": self.n_representatives,
            "compression": self.compression,
            "linkage": self.linkage,
            "sample_size": self.sample_size,
            "seed": self.seed,
            "pruning_fraction": self.pruning_fraction,
            "pruning_min_size": self.pruning_min_size,
            "assignment_threshold": self.assignment_threshold,
            "n_samples": self.agg_.n_samples_,
            "noise_points": self.agg_.noise_points_
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(
            save_path, 
            linkage_matrix=self.agg_.linkage_matrix_,
            sample_data=self.X_sample_,
            meta_data=meta_data
        )

    @classmethod
    def from_pretrained(cls, model_path):
        """
        Loads a pretrained CURE model from an .npz file.
        
        Parameters:
        -----------
        model_path : str
            Path to the .npz file containing 'linkage_matrix' and 'sample_data'.
        """
        data = np.load(model_path, allow_pickle=True)
        linkage_matrix = data['linkage_matrix']
        sample_data = data['sample_data']
        meta_data = data['meta_data'].item()
        
        # Create instance
        instance = cls(sample_size=meta_data["sample_size"],
                       n_representatives=meta_data["n_representatives"],
                       compression=meta_data["compression"],
                       linkage=meta_data["linkage"],
                       pruning_fraction=meta_data["pruning_fraction"],
                       pruning_min_size=meta_data["pruning_min_size"],
                       assignment_threshold=meta_data["assignment_threshold"],
                       seed=meta_data["seed"])
        
        # Restore state
        instance.X_sample_ = sample_data
        instance.agg_ = AgglomerativeClustering(linkage=meta_data["linkage"], pruning_fraction=meta_data["pruning_fraction"], pruning_min_size=meta_data["pruning_min_size"])
        instance.agg_.linkage_matrix_ = linkage_matrix
        instance.agg_.n_samples_ = meta_data["n_samples"]
        instance.agg_.noise_points_ = meta_data["noise_points"]
        
        return instance

    def _get_representatives(self, n_clusters):
        """
        Internal helper to calculate representatives for a specific number of clusters.
        """
        if self.agg_ is None:
            raise RuntimeError("Model not fitted.")

        # 1. Get sample labels from the hierarchical tree
        sample_labels = self.agg_.get_labels(n_clusters)

        # 2. Select Representatives
        representatives = []
        
        for k in range(n_clusters):
            # Get points in this cluster
            cluster_points = self.X_sample_[sample_labels == k]
            
            if len(cluster_points) == 0:
                representatives.append([])
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
                dists = [euclidean_distance(p, centroid) for p in cluster_points]
                first_idx = np.argmax(dists)
                reps.append(cluster_points[first_idx])
                
                # 2. Subsequent points farthest from existing reps
                for _ in range(self.n_representatives - 1):
                    max_min_dist = -1
                    best_candidate = None
                    
                    for p in cluster_points:
                        # Find min distance to any existing rep
                        min_dist_to_reps = min([euclidean_distance(p, r) for r in reps])
                        
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
            
        return representatives
    
    def predict(self, X, n_clusters):
        """
        Assigns labels to new data points based on the fitted model and a specific number of clusters.
        """
        if self.agg_ is None:
            raise RuntimeError("Run fit() or load_pretrained() before predict()")
        
        X = np.array(X)
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        # Get representatives for the requested number of clusters
        representatives = self._get_representatives(n_clusters)

        # Assign entire dataset (Linear Scan)
        for i in range(n_samples):
            point = X[i]
            min_dist = np.inf
            best_cluster = -1
            
            # Check distance to ALL representatives of ALL clusters
            for cluster_idx, reps in enumerate(representatives):
                if not reps:
                    continue
                for r in reps:
                    dist = euclidean_distance(point, r)
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = cluster_idx
            
            if min_dist > self.assignment_threshold:
                labels[i] = -1
            else:
                labels[i] = best_cluster
            
        return labels

    def get_labels(self, n_clusters):
        """
        Generates labels for the full dataset used during fit().
        """
        if self.X_ is None:
            raise RuntimeError("No training data available (model might be loaded from disk). Use predict(X, n_clusters) instead.")
            
        return self.predict(self.X_, n_clusters)
    
    def plot_dendrogram(self, p=100, show_plot=False, save_path: str | None = None, **kwargs):
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
        
        self.agg_.plot_dendrogram(p=p, show_plot=show_plot, save_path=save_path, **kwargs)
    
if __name__ == "__main__":
    parser = ArgumentParser("CURE Hierarchical Clustering for Large Datasets. Build and save entire tree.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset (embeddings).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the CURE model and plots.")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of points to sample for clustering.")
    parser.add_argument("--n_representatives", type=int, default=20, help="Number of representatives per cluster.")
    parser.add_argument("--compression", type=float, default=0.2, help="Compression factor towards centroid (alpha).")
    parser.add_argument("--linkage", type=str, default="average", choices=["single", "average", "ward"], help="Linkage criterion for hierarchical clustering.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--pruning_fraction", type=float, default=0.05, help="Fraction of clusters to prune as noise during fitting.")
    parser.add_argument("--pruning_min_size", type=int, default=1, help=" Minimum cluster size to avoid being pruned as noise.")
    parser.add_argument("--assignment_threshold", type=float, default=None, help="Maximum distance to assign a point to a cluster.")
    parser.add_argument("--dendrogram_p", type=int, default=100, help="Number of last merges to show in dendrogram plot.")
    args = parser.parse_args()
    
    logger = CustomLogger(project_name='Computational-Tools', group='clustering', run_name='train_cure_final', use_wandb=True)
    
    logger.log_config(vars(args))
    
    logger.info("Loading embeddings...")
    data = np.load(args.data_path)
    embeddings = data["embeddings"]
    del data
    logger.info(f"Loaded {embeddings.shape[0]} samples with dimension {embeddings.shape[1]} from {args.data_path}\n")
    
    logger.info("Fitting CURE hierarchical clustering...")
    cure = CURE(sample_size=args.sample_size, n_representatives=args.n_representatives, compression=args.compression, linkage=args.linkage, assignment_threshold=args.assignment_threshold)
    cure.fit(embeddings)
    logger.info("Fitted CURE model.\n")
    
    logger.info(f"Saving linkage matrix and sample data to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    model_path = os.path.join(args.output_path, "cure_model.npz")
    cure.save(model_path)
    logger.artifact(model_path, "cure_model_test", "model")
    
    logger.info("Saving dendrogram plot...")
    dendrogram_path = os.path.join(args.output_path, "dendrogram.png")
    cure.plot_dendrogram(save_path=dendrogram_path, p=args.dendrogram_p)
    logger.artifact(dendrogram_path, "dendrogram", "image")
    
    # Interactive dendrogram
    #logger.info("Saving interactive dendrogram plot...")
    #interactive_dendrogram_path = os.path.join(args.output_path, "interactive_dendrogram.html")
    #plot_interactive_dendrogram(cure, save_path=interactive_dendrogram_path, p=args.dendrogram_p)
    #logger.artifact(interactive_dendrogram_path, "interactive_dendrogram", "html")
    
    scores = []
    pct_noise = []
    for n_clusters in [2, 5, 10, 20]:
        try:
            labels = cure.predict(embeddings, n_clusters=n_clusters)
            non_noise_indeces = labels != -1
            score = davies_bouldin_index(embeddings[non_noise_indeces], labels[non_noise_indeces])
            scores.append(score)
            pct_noise.append(len(labels[labels == -1]) / len(labels) * 100)
        except Exception as e:
            logger.warning(f"Could not compute DBI for {n_clusters} clusters: {e}")
            scores.append(float('nan'))
            pct_noise.append(float('nan'))
    
    logger.info("Davies-Bouldin Index scores for different cluster counts:")
    for n_clusters, score in zip([2, 5, 10, 20], scores):
        pct_noise_for_n_clusters = pct_noise[[2,5,10,20].index(n_clusters)]
        logger.info(f" - {n_clusters} clusters: DBI = {score:.4f}, Noise% = {pct_noise_for_n_clusters:.2f}%")
        logger.log_summary({f"DBI_{n_clusters}_clusters": score, f"PCT_Noise_{n_clusters}_clusters": pct_noise_for_n_clusters})
    
    logger.info("Done.")