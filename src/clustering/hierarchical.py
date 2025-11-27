import numpy as np
from argparse import ArgumentParser
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from src.clustering.utils import davies_bouldin_index
from src.utils.logging import CustomLogger
import matplotlib.colors as mcolors
from src.clustering.interactive_plot import plot_interactive_dendrogram
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
    def __init__(self, linkage='centroid', pruning_trigger_fraction=0.33, phase1_cutoff_count=1):
        self.linkage = linkage
        self.pruning_trigger_fraction = pruning_trigger_fraction
        self.phase1_cutoff_count = phase1_cutoff_count
        self.linkage_matrix_ = None # Stores the (n-1) x 4 linkage matrix for dendrograms
        self.pruned_indices_ = set()
        self.n_samples_ = 0
        self.n_real_merges_ = 0

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
        self.n_samples_ = X.shape[0]
        n_samples = self.n_samples_
        
        # Dictionary to store active clusters: {cluster_id: [list of sample indices]}
        # We start with clusters 0 to n-1
        current_clusters = {i: [i] for i in range(n_samples)}
        
        # List to store the history of merges in Linkage Matrix format
        # [idx1, idx2, distance, sample_count]
        self.linkage_matrix_ = []
        self.pruned_indices_ = set()
        
        # Track pruned clusters for "Ghost Merge"
        pruned_clusters = [] # List of (cluster_id, size)
        
        # Counter for the next cluster ID (starts after the last sample index)
        next_cluster_id = n_samples

        pruning_triggered = False

        # Loop until one cluster remains
        while len(current_clusters) > 1:
            # Phase 1 Pruning
            if not pruning_triggered and len(current_clusters) <= n_samples * self.pruning_trigger_fraction:
                pruning_triggered = True
                ids_to_prune = []
                for cid, indices in current_clusters.items():
                    if len(indices) <= self.phase1_cutoff_count:
                        ids_to_prune.append(cid)
                
                for cid in ids_to_prune:
                    self.pruned_indices_.update(current_clusters[cid])
                    pruned_clusters.append((cid, len(current_clusters[cid])))
                    del current_clusters[cid]
                
                if len(current_clusters) < 2:
                    break

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

        # Save the number of real merges before adding ghost merges
        self.n_real_merges_ = len(self.linkage_matrix_)

        # --- Ghost Merge Step ---
        # If we have pruned clusters, we need to merge them back to complete the tree.
        if pruned_clusters:
            # Determine max distance for ghost merges
            if len(self.linkage_matrix_) > 0:
                # linkage_matrix_ is a list here
                max_real_dist = max(row[2] for row in self.linkage_matrix_)
                ghost_dist = max_real_dist * 1.5
            else:
                ghost_dist = 1.0 # Default if no merges happened yet
                
            # 1. Merge all pruned clusters into a "Noise Super-Cluster"
            # We treat the first pruned cluster as the accumulator
            noise_cid, noise_size = pruned_clusters[0]
            
            for i in range(1, len(pruned_clusters)):
                next_cid, next_size = pruned_clusters[i]
                
                # Merge
                new_size = noise_size + next_size
                self.linkage_matrix_.append([float(noise_cid), float(next_cid), float(ghost_dist), float(new_size)])
                
                noise_cid = next_cluster_id
                noise_size = new_size
                next_cluster_id += 1
                
            # 2. Merge the Noise Super-Cluster with the remaining valid cluster(s)
            # current_clusters should have 0 or 1 item usually.
            valid_clusters_info = [(cid, len(indices)) for cid, indices in current_clusters.items()]
            
            # Add the noise super cluster to the list
            valid_clusters_info.append((noise_cid, noise_size))
            
            # Merge everything remaining
            while len(valid_clusters_info) > 1:
                c1_id, c1_size = valid_clusters_info.pop(0)
                c2_id, c2_size = valid_clusters_info.pop(0)
                
                new_size = c1_size + c2_size
                self.linkage_matrix_.append([float(c1_id), float(c2_id), float(ghost_dist), float(new_size)])
                
                # The new cluster ID is next_cluster_id
                new_id = next_cluster_id
                next_cluster_id += 1
                
                # Insert back to merge with others if any
                valid_clusters_info.insert(0, (new_id, new_size))

        self.linkage_matrix_ = np.array(self.linkage_matrix_)
        return self

    def get_labels(self, n_clusters):
        """
        Replays the clustering history to return labels for a specific number of clusters.
        """
        if self.linkage_matrix_ is None:
            raise RuntimeError("Model must be fit before getting labels.")
            
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be at least 1")

        # Reconstruct state
        # 1. Start with singletons
        current_clusters = {i: [i] for i in range(self.n_samples_)}
        
        # 2. We perform merges to reach the desired state
        # We only care about REAL merges, not ghost merges.
        # Active samples are those NOT pruned in Phase 1.
        n_active_samples = self.n_samples_ - len(self.pruned_indices_)
        
        # If we want n_clusters valid clusters, we need to reduce n_active_samples to n_clusters.
        # Number of merges needed = n_active_samples - n_clusters.
        merges_needed = n_active_samples - n_clusters
        
        if merges_needed < 0:
             merges_needed = 0
             
        # Safety check: ensure we don't exceed real merges
        if merges_needed > self.n_real_merges_:
            merges_needed = self.n_real_merges_
        
        for i in range(merges_needed):
            # linkage_matrix row: [c1, c2, dist, size]
            row = self.linkage_matrix_[i]
            c1, c2 = int(row[0]), int(row[1])
            new_id = self.n_samples_ + i # Note: IDs in linkage matrix are consistent with this
            
            # Merge
            if c1 in current_clusters and c2 in current_clusters:
                current_clusters[new_id] = current_clusters[c1] + current_clusters[c2]
                del current_clusters[c1]
                del current_clusters[c2]
            else:
                # This should not happen if logic is correct
                pass

        # 3. Assign labels
        labels = np.full(self.n_samples_, -1, dtype=int)
        
        # Sort keys to ensure deterministic labeling order
        # We only assign non-negative labels to clusters that are NOT pruned.
        valid_label_id = 0
        for cluster_id in sorted(current_clusters.keys()):
            indices = current_clusters[cluster_id]
            
            # Check if this cluster is pruned
            # If any point in the cluster is in pruned_indices_, then the whole cluster is pruned
            # (because pruned points are never merged with non-pruned ones until the ghost merges, which we skipped)
            is_pruned = False
            if len(indices) > 0:
                # Just check the first point, as clusters are either all pruned or all valid
                if indices[0] in self.pruned_indices_:
                    is_pruned = True
            
            if not is_pruned:
                for sample_idx in indices:
                    labels[sample_idx] = valid_label_id
                valid_label_id += 1
            # Else leave as -1
                
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

        # --- Feature 1: Coloring by Cluster Size & Ghost Merges ---
        link_color_func = None
        
        # Determine ghost threshold
        distances = self.linkage_matrix_[:, 2]
        max_dist = np.max(distances)
        has_pruned = len(self.pruned_indices_) > 0
        # If we have pruned points, the max distance is likely the ghost distance
        ghost_threshold = max_dist * 0.99 if has_pruned else np.inf

        if color_by_size:
            n_samples = len(self.linkage_matrix_) + 1
            cluster_sizes = {i: 1 for i in range(n_samples)}
            
            for i, row in enumerate(self.linkage_matrix_):
                cluster_idx = n_samples + i
                cluster_sizes[cluster_idx] = int(row[3])

            cmap = plt.get_cmap('viridis') 
            norm = mcolors.LogNorm(vmin=1, vmax=n_samples)

            def size_color_func(k):
                # Check for ghost merge
                if k >= n_samples:
                    idx = k - n_samples
                    if self.linkage_matrix_[idx, 2] >= ghost_threshold:
                        return 'gray'

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
    def __init__(self, sample_size=200, n_representatives=4, compression=0.5, linkage='centroid',
                 phase2_min_ratio=0.05, assignment_threshold=np.inf,
                 pruning_trigger_fraction=0.33, phase1_cutoff_count=1, seed=42):
        self.sample_size = sample_size
        self.n_representatives = n_representatives
        self.compression = compression # Alpha: how much to shrink towards centroid
        self.linkage = linkage
        self.phase2_min_ratio = phase2_min_ratio
        self.assignment_threshold = assignment_threshold
        self.pruning_trigger_fraction = pruning_trigger_fraction
        self.phase1_cutoff_count = phase1_cutoff_count
        self.seed = seed
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
            np.random.seed(self.seed)
            sample_indices = np.random.choice(n_samples, self.sample_size, replace=False)
            self.X_sample_ = self.X_[sample_indices]

        # 2. Cluster the Sample (The expensive part, but on small N)
        # We reuse our custom AgglomerativeClustering class which builds the full tree
        self.agg_ = AgglomerativeClustering(linkage=self.linkage,
                                            pruning_trigger_fraction=self.pruning_trigger_fraction,
                                            phase1_cutoff_count=self.phase1_cutoff_count)
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
            "phase2_min_ratio": self.phase2_min_ratio,
            "assignment_threshold": self.assignment_threshold,
            "pruning_trigger_fraction": self.pruning_trigger_fraction,
            "phase1_cutoff_count": self.phase1_cutoff_count
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
                       phase2_min_ratio=meta_data.get("phase2_min_ratio", 0.05),
                       assignment_threshold=meta_data.get("assignment_threshold", np.inf),
                       pruning_trigger_fraction=meta_data.get("pruning_trigger_fraction", 0.33),
                       phase1_cutoff_count=meta_data.get("phase1_cutoff_count", 1))
        
        # Restore state
        instance.X_sample_ = sample_data
        instance.agg_ = AgglomerativeClustering(linkage=meta_data["linkage"],
                                                pruning_trigger_fraction=instance.pruning_trigger_fraction,
                                                phase1_cutoff_count=instance.phase1_cutoff_count)
        instance.agg_.linkage_matrix_ = linkage_matrix
        instance.agg_.n_samples_ = len(sample_data) # Restore n_samples_
        # Note: pruned_indices_ are not saved in current format. 
        # If we need them, we should save them. 
        # However, for prediction, we only need representatives.
        # Representatives are calculated from sample_labels.
        # sample_labels are calculated from linkage_matrix.
        # If pruned_indices_ are missing, get_labels might behave differently if we rely on them.
        # But wait, get_labels relies on pruned_indices_ to assign -1.
        # If we don't restore pruned_indices_, get_labels will assign valid labels to pruned points (if they exist in linkage?).
        # No, pruned points are NOT in linkage matrix.
        # So get_labels will see them as singletons (orphans).
        # And since they are not in pruned_indices_, it will assign them a valid label?
        # Yes, it will assign them a new label ID.
        # This might be an issue if we want to reproduce exact state.
        # But usually we just use predict() which uses representatives.
        # Representatives calculation uses get_labels.
        # If get_labels returns a valid label for a pruned point, it will be included in that cluster.
        # But since it's a singleton, it will form a small cluster.
        # Then Phase 2 pruning (check size) will likely catch it and mark it as noise.
        # So it might be fine.
        
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
        representatives = {}
        
        # Note: get_labels might return -1 for Phase 1 pruned points.
        # We should ignore them.
        
        unique_labels = np.unique(sample_labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        for k in valid_labels:
            # Get points in this cluster
            cluster_points = self.X_sample_[sample_labels == k]
            
            if len(cluster_points) == 0:
                continue

            # Phase 2 Pruning
            if len(cluster_points) < self.sample_size * self.phase2_min_ratio:
                # Noise Group - leave representatives empty
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
            
            representatives[k] = shrunk_reps
            
        return representatives
    
    def predict(self, X, n_clusters, assignment_threshold=None):
        """
        Assigns labels to new data points based on the fitted model and a specific number of clusters.
        """
        if self.agg_ is None:
            raise RuntimeError("Run fit() or load_pretrained() before predict()")
        
        if assignment_threshold is None:
            assignment_threshold = self.assignment_threshold
        
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
            for cluster_idx, reps in representatives.items():
                if not reps:
                    continue
                for r in reps:
                    dist = self._euclidean_distance(point, r)
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = cluster_idx
            
            if min_dist > assignment_threshold:
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
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the CURE model and plots.")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of points to sample for clustering.")
    parser.add_argument("--n_representatives", type=int, default=20, help="Number of representatives per cluster.")
    parser.add_argument("--compression", type=float, default=0.4, help="Compression factor towards centroid (alpha).")
    parser.add_argument("--linkage", type=str, default="single", choices=["centroid", "single", "complete", "average"], help="Linkage criterion for hierarchical clustering.")
    parser.add_argument("--dendrogram_p", type=int, default=100, help="Number of last merges to show in dendrogram plot.")
    
    # New arguments for noise handling
    parser.add_argument("--pruning_trigger", type=float, default=0.05, help="Fraction of clusters remaining to trigger Phase 1 pruning.")
    parser.add_argument("--phase1_cutoff", type=int, default=1, help="Max cluster size to prune in Phase 1.")
    parser.add_argument("--phase2_ratio", type=float, default=0.01, help="Min ratio of sample size for a cluster to be valid in Phase 2.")
    parser.add_argument("--assignment_threshold", type=float, default=np.inf, help="Max distance to assign a point to a cluster.")

    args = parser.parse_args()
    
    logger = CustomLogger(project_name='Computational-Tools', group='clustering', run_name='tune_cure_fine', use_wandb=True)
    
    logger.log_config(vars(args))
    
    logger.info("Loading embeddings...")
    data = np.load(args.data_path)
    embeddings = data["embeddings"]
    del data
    logger.info(f"Loaded {embeddings.shape[0]} samples with dimension {embeddings.shape[1]} from {args.data_path}\n")
    
    logger.info("Fitting CURE hierarchical clustering...")
    cure = CURE(sample_size=args.sample_size, 
                n_representatives=args.n_representatives, 
                compression=args.compression, 
                linkage=args.linkage,
                pruning_trigger_fraction=args.pruning_trigger,
                phase1_cutoff_count=args.phase1_cutoff,
                phase2_min_ratio=args.phase2_ratio,
                assignment_threshold=args.assignment_threshold)
    cure.fit(embeddings)
    logger.info("Fitted CURE model.\n")
    
    logger.info(f"Saving linkage matrix and sample data to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    model_path = os.path.join(args.output_path, "cure_model.npz")
    cure.save(model_path)
    logger.artifact(model_path, "cure_model", "model")
    
    logger.info("Saving dendrogram plot...")
    dendrogram_path = os.path.join(args.output_path, "dendrogram.png")
    cure.plot_dendrogram(save_path=dendrogram_path, p=args.dendrogram_p)
    logger.artifact(dendrogram_path, "dendrogram", "image")
    
    # Interactive dendrogram
    logger.info("Saving interactive dendrogram plot...")
    interactive_dendrogram_path = os.path.join(args.output_path, "interactive_dendrogram.html")
    plot_interactive_dendrogram(cure, save_path=interactive_dendrogram_path, p=args.dendrogram_p)
    logger.artifact(interactive_dendrogram_path, "interactive_dendrogram", "html")
    
    results = {}
    search_n_clusters = [2, 5, 10, 20, 50, 100]
    for n_clusters in search_n_clusters:
        logger.info(f"Evaluating for {n_clusters} clusters...")
        labels = cure.predict(embeddings, n_clusters=n_clusters)
        valid_mask = labels != -1
        
        unique_labels = np.unique(labels[valid_mask])
        if len(unique_labels) < 2:
            logger.warning(f" - {n_clusters} clusters: Skipped (Not enough valid clusters found after noise removal)")
            continue
        
        score = davies_bouldin_index(embeddings[valid_mask], labels[valid_mask])
        
        pct_noise = 1.0 - (np.sum(valid_mask) / len(labels))
        results[n_clusters] = (score, pct_noise)
    
    logger.info("Davies-Bouldin Index scores for different cluster counts:")
    for n_clusters, result in results.items():
        score, pct_noise = result
        logger.info(f" - {n_clusters} clusters: DBI = {score:.4f}, Noise% = {pct_noise*100:.2f}%")
        logger.log_summary({f"DBI_{n_clusters}_clusters": score, f"Noise_Percent_{n_clusters}_clusters": pct_noise})
        
    logger.info("Done.")