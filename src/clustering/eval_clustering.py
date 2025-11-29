import os
import pickle
import shutil
import joblib
import time
import numpy as np
import torch
import umap
import pandas as pd
import warnings
from tqdm.auto import tqdm
import seaborn as sns
from argparse import ArgumentParser
from src.utils.logging import CustomLogger
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
warnings.filterwarnings("ignore", message=".*pin_memory.*")

from src.sequence_modelling.models import AISBERT
from src.clustering.utils import davies_bouldin_index
from src.utils.datasets import AISDataset

def pickle_and_artifact(obj, path: str, logger: CustomLogger = None):
    """Pickle object and log artifact if logger is provided."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    if logger:
        logger.artifact(path, name=os.path.basename(path), type="pickle")
        
def save_plot_and_artifact(fig, path: str, logger: CustomLogger = None):
    """Save matplotlib figure and log artifact if logger is provided."""
    fig.savefig(path)
    if logger:
        logger.artifact(path, name=os.path.basename(path), type="plot")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory containing processed data.')
    parser.add_argument('--output_dir', type=str, required=False, default='data/clustering_results',
                        help='Path to output directory to save clustering results.')
    parser.add_argument('--include_noise', action='store_true', default=False,
                        help='Whether to include noise points in evaluation metrics.')
    args = parser.parse_args()
    np.random.seed(42)
    
    logger = CustomLogger(project_name='Computational-Tools', group='eval', run_name='run_eval', use_wandb=True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    # Get model and dataset
    model = AISBERT.from_pretrained('models/pretrained_models/pretrained_bert-3026')

    processed_dir = args.data_dir
    ds = AISDataset(data_dir=processed_dir)
    try:
        vessel_types_mapping = joblib.load(f) # dict
    except:
        with open(os.path.join(processed_dir, "vessel_types.pkl"), "rb") as f:
            vessel_types_mapping = pickle.load(f) # dict
            
    from src.sequence_modelling.utils import get_embeddings
    embeddings, mmsis, start_times = get_embeddings(ds, model, l2_normalize=True)
            
    #### UMAP 2D Visualization ####
    try:
        reducer = umap.UMAP(n_components=2, random_state=42)
        
        # Pick 5000 random samples for UMAP visualization
        np.random.seed(42)
        sample_indices = np.random.choice(len(ds), size=3000, replace=False)
        embeddings_umap = reducer.fit_transform(embeddings[sample_indices])
        
        # Save UMAP embeddings
        umap_output_path = os.path.join(args.output_dir, "umap_embeddings.pkl")
        pickle_and_artifact({
            'embeddings_umap': embeddings_umap,
            'sample_indices': sample_indices
        }, umap_output_path, logger)
        logger.info(f"Saved UMAP embeddings to {umap_output_path}")
        
        # Plot UMAP
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], c='blue', alpha=0.5)
        plt.title("UMAP of AIS Embeddings")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.grid(True)
        save_plot_and_artifact(plt, os.path.join(args.output_dir, 'plots', 'umap_plot.png'), logger)
        plt.close()
        logger.info(f"Saved UMAP plot to {os.path.join(args.output_dir, 'plots', 'umap_plot.png')}")
    except Exception as e:
        logger.warning(f"UMAP visualization failed: {e}")
    
    #### Perform Clustering
    # Load pretrained
    from src.clustering.hierarchical import CURE
    cure_model = CURE.from_pretrained('models/pretrained_models/cure_model.npz')
    
    def get_track_from_index(index, ds, mmsis, start_times):
        mmsi = mmsis[index]
        start_time = start_times[index]
        track, _,_,_ = ds.get_sample_by_mmsi_and_start_time(mmsi, start_time)
        return track
    
    labels_2 = cure_model.predict(embeddings, n_clusters=2)
    labels_3 = cure_model.predict(embeddings, n_clusters=3)
    labels_5 = cure_model.predict(embeddings, n_clusters=5)
    
    try:
    
        dav_boild_idx_2 = davies_bouldin_index(embeddings, labels_2)
        logger.info(f"CURE with 2 clusters: Davies-Bouldin Index = {dav_boild_idx_2:.4f}")

        dav_boild_idx_3 = davies_bouldin_index(embeddings, labels_3)
        logger.info(f"CURE with 3 clusters: Davies-Bouldin Index = {dav_boild_idx_3:.4f}")
        
        dav_boild_idx_5 = davies_bouldin_index(embeddings, labels_5)
        logger.info(f"CURE with 5 clusters: Davies-Bouldin Index = {dav_boild_idx_5:.4f}")
        
        # pct noise
        pct_noise_2 = np.sum(labels_2 == -1) / len(labels_2) * 100
        pct_noise_3 = np.sum(labels_3 == -1) / len(labels_3) * 100
        pct_noise_5 = np.sum(labels_5 == -1) / len(labels_5) * 100
        logger.info(f"CURE with 2 clusters: Percentage of Noise = {pct_noise_2:.2f}%")
        logger.info(f"CURE with 3 clusters: Percentage of Noise = {pct_noise_3:.2f}%")
        logger.info(f"CURE with 5 clusters: Percentage of Noise = {pct_noise_5:.2f}%")
        
        # Save clustering results
        clustering_output_path = os.path.join(args.output_dir, "clustering_results.pkl")
        pickle_and_artifact({
                'labels_2': labels_2,
                'labels_3': labels_3,
                'labels_5': labels_5,
                'davies_bouldin_index': {
                    '2_clusters': dav_boild_idx_2,
                    '3_clusters': dav_boild_idx_3,
                    '5_clusters': dav_boild_idx_5
                },
                'pct_noise': {
                    '2_clusters': pct_noise_2,
                    '3_clusters': pct_noise_3,
                    '5_clusters': pct_noise_5
                }
            }, clustering_output_path, logger)
        logger.log_summary({
            "DBI_2_clusters": dav_boild_idx_2,
            "DBI_3_clusters": dav_boild_idx_3,
            "DBI_5_clusters": dav_boild_idx_5,
            "Pct_Noise_2_clusters": pct_noise_2,
            "Pct_Noise_3_clusters": pct_noise_3,
            "Pct_Noise_5_clusters": pct_noise_5
        })
    
        logger.info(f"Saved clustering results to {clustering_output_path}")
    except Exception as e:
        logger.warning(f"Clustering eval failed: {e}")
    
    try:
        np.random.seed(42)
        sample_indices = np.random.choice(len(ds), size=3000, replace=False)
        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_umap = reducer.fit_transform(embeddings[sample_indices])
        
        # Save UMAP embeddings for clustered samples
        umap_clustered_output_path = os.path.join(args.output_dir, "umap_clustered_embeddings.pkl")
        pickle_and_artifact({
                'embeddings_umap': embeddings_umap,
                'sample_indices': sample_indices,
                'labels_2': labels_2[sample_indices],
                'labels_3': labels_3[sample_indices],
                'labels_5': labels_5[sample_indices]
            }, umap_clustered_output_path, logger)
        logger.info(f"Saved UMAP clustered embeddings to {umap_clustered_output_path}")
        
        from matplotlib import colormaps as cm
        import matplotlib.colors as mcolors
        for cluster_size, labels in [(2, labels_2), (3, labels_3), (5, labels_5)]:
            plt.figure(figsize=(10, 8))
            
            # 1. Identify unique labels and count them
            unique_labels = sorted(np.unique(labels))
            n_labels = len(unique_labels)
            
            # 2. Create the color palette
            # Get the tab10 colormap
            base_cmap = cm.get_cmap('tab10')
            color_list = [base_cmap(i) for i in range(n_labels)]
            
            # Check if -1 exists in labels
            if -1 in unique_labels:
                # Find the index of -1 and replace that color with Grey
                neg_index = unique_labels.index(-1)
                color_list[neg_index] = (0.5, 0.5, 0.5, 1.0)  # Grey color
                
            # Create a custom ListedColormap
            cmap_discrete = mcolors.ListedColormap(color_list)
            
            # 3. Create a Norm (maps specific integer values to specific colors)
            # We define boundaries between integers (e.g., -1.5, -0.5, 0.5, 1.5...)
            bounds = np.arange(min(unique_labels) - 0.5, max(unique_labels) + 1.5, 1)
            norm = mcolors.BoundaryNorm(bounds, cmap_discrete.N)

            # 4. Plot
            scatter = plt.scatter(
                embeddings_umap[:, 0], 
                embeddings_umap[:, 1], 
                c=labels, 
                cmap=cmap_discrete, 
                norm=norm, # Important: Apply the norm!
                s=10
            )
            
            plt.title(f"CURE Clustering with {cluster_size} Clusters")
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")
            
            # 5. Fix the Colorbar ticks to center on the integers
            cbar = plt.colorbar(scatter, ticks=unique_labels)
            cbar.set_label('Cluster Label')
            
            save_plot_and_artifact(plt, os.path.join(args.output_dir, 'plots', f'umap_cure_{cluster_size}_clusters.png'), logger)
            plt.close()
            logger.info(f"Saved UMAP CURE {cluster_size} clusters plot to {os.path.join(args.output_dir, 'plots', f'umap_cure_{cluster_size}_clusters.png')}")
    except Exception as e:
        logger.warning(f"UMAP clustered visualization failed: {e}")
    
    try:
        from src.utils.plot import plot_trajectories
        from matplotlib import colormaps as cm

        # Plot 200 random trajectories colored by cluster labels
        color_map = cm.get_cmap('tab10').colors
        color_map = list(color_map)
        color_map.insert(0, (0.5, 0.5, 0.5))  # Grey for noise

        np.random.seed(42)
        random_indices = np.random.choice(len(mmsis), size=200, replace=False)
        for cluster_size, labels in [(2, labels_2), (3, labels_3), (5, labels_5)]:
            tracks_to_plot = []
            for i in random_indices:
                track = get_track_from_index(i, ds, mmsis, start_times)
                cluster_label = labels[i]
                color = color_map[cluster_label + 1]  # +1 to account for noise color at index 0
                tracks_to_plot.append({'track': track, 'color': color, 'label': cluster_label})
            pickle_and_artifact(tracks_to_plot, os.path.join(args.output_dir, f'trajectories_cure_{cluster_size}_clusters.pkl'), logger)
            fig = plot_trajectories(tracks_to_plot, show_plot=False)
            save_plot_and_artifact(fig, os.path.join(args.output_dir, 'plots', f'trajectories_cure_{cluster_size}_clusters.png'), logger)
            plt.close(fig)
            logger.info(f"Saved Trajectories CURE {cluster_size} clusters plot to {os.path.join(args.output_dir, 'plots', f'trajectories_cure_{cluster_size}_clusters.png')}")
    except Exception as e:
        logger.warning(f"Trajectory plotting failed: {e}")
        
    data = []
    for i in range(len(ds)):
        row = ds.get_sample_features(i)
        row['vessel_type'] = vessel_types_mapping.get(int(mmsis[i]), 59)
        row['cluster_label_2'] = labels_2[i]
        row['cluster_label_3'] = labels_3[i]
        row['cluster_label_5'] = labels_5[i]
        data.append(row)
    df = pd.DataFrame(data)
    metadata_output_path = os.path.join(args.output_dir, "clustering_metadata_features.pkl")
    pickle_and_artifact(df, metadata_output_path, logger)
    
    try:
        continuous_features = ['speed_avg', 'speed_std', 'speed_max', 'duration', 'lat_max', 'lat_min', 'lon_max', 'lon_min', 'displacement', 'length', 'length_over_displacement', 'cog_std']
        scores_df = (df[continuous_features] - df[continuous_features].mean()) / df[continuous_features].std()

        feature_importances = {f: [] for f in continuous_features}
        for cluster_size in [2, 3, 5]:
            cluster_label_col = f'cluster_label_{cluster_size}'
            scores_df_ = scores_df.copy()
            scores_df_[cluster_label_col] = df[cluster_label_col]
            scores_df_ = scores_df_[scores_df_[cluster_label_col] != -1] if not args.include_noise else scores_df_

            cluster_profiles = scores_df_.groupby(cluster_label_col).mean()
            
            # Calculate standard deviation across clusters for each feature
            feature_importance = cluster_profiles.std().sort_values(ascending=False)
            for f in continuous_features:
                feature_importances[f].append(feature_importance[f])
            
            pickle_and_artifact(cluster_profiles, os.path.join(args.output_dir, f'cluster_profiles_cure_{cluster_size}_clusters.pkl'), logger)
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feature_importance.values, y=feature_importance.index, palette='viridis')
            plt.xlim(0,1)
            plt.title(f"Feature Importance (Std Dev across Clusters) - CURE with {cluster_size} Clusters")
            plt.xlabel("Standard Deviation")
            plt.ylabel("Feature")
            save_plot_and_artifact(plt, os.path.join(args.output_dir, 'plots', f'feature_importance_cure_{cluster_size}_clusters.png'), logger)
            plt.close()
            logger.info(f"Saved Feature Importance plot for CURE {cluster_size} clusters to {os.path.join(args.output_dir, 'plots', f'feature_importance_cure_{cluster_size}_clusters.png')}")
    
        max_importance_per_feature = {f: max(importances) for f, importances in feature_importances.items()}
        sorted_features = sorted(max_importance_per_feature.items(), key=lambda x: x[1], reverse=True)
        logger.info("Feature importances (max std dev across clusterings):")
        for feature, importance in sorted_features:
            logger.info(f"{feature}: {importance:.4f}")
        pickle_and_artifact(feature_importances, os.path.join(args.output_dir, 'feature_importances_overall.pkl'), logger)
    except Exception as e:
        logger.warning(f"Feature importance calculation failed: {e}")
        
    try:
        continuous_features = [f for f in continuous_features if f not in ['speed_max', 'speed_std', 'duration', 'length_over_displacement']] # Low importance
        scores_df = (df[continuous_features] - df[continuous_features].mean()) / df[continuous_features].std()

        for cluster_size in [2, 3, 5]:
            cluster_label_col = f'cluster_label_{cluster_size}'
            scores_df_ = scores_df.copy()
            scores_df_[cluster_label_col] = df[cluster_label_col]
            scores_df_ = scores_df_[scores_df_[cluster_label_col] != -1] if not args.include_noise else scores_df_

            cluster_profiles = scores_df_.groupby(cluster_label_col).mean()
            
            pickle_and_artifact(cluster_profiles, os.path.join(args.output_dir, f'cluster_profiles_cure_{cluster_size}_clusters_reduced.pkl'), logger)

            plt.figure(figsize=(12, 6))
            sns.heatmap(
                cluster_profiles, 
                annot=True,
                cmap='RdBu_r',
                center=0,
                fmt='.2f',
                vmax=2,
                vmin=-2,
                yticklabels=[f'C{i} (n={len(df[df[cluster_label_col]==i])})' for i in cluster_profiles.index]
            )
            plt.title(f"Cluster Characteristics (Z-scores) - CURE with {cluster_size} Clusters")
            save_plot_and_artifact(plt, os.path.join(args.output_dir, 'plots', f'cluster_characteristics_cure_{cluster_size}_clusters.png'), logger)
            plt.close()
            logger.info(f"Saved Cluster Characteristics heatmap for CURE {cluster_size} clusters to {os.path.join(args.output_dir, 'plots', f'cluster_characteristics_cure_{cluster_size}_clusters.png')}")
            
    except Exception as e:
        logger.warning(f"Cluster characteristics plotting failed: {e}")
    
    # Simplify vessel types to major categories
    c, t, f, s = 'Cargo', 'Tanker', 'Fishing', 'Sailing/Pleasure'
    vt_adapter = {
        70: c, 71: c, 72: c, 73: c, 74: c, 75: c, 76: c, 77: c, 78: c, # Cargo
        80: t, 81: t, 82: t, 83: t, 84: t, 85: t, 86: t, 87: t, 88: t, # Tanker
        30: f, # Fishing
        37: s, 36: s # Sailing / Pleasure
    }
    df['vessel_type_simple'] = df['vessel_type'].map(vt_adapter).fillna("Other")
    
    # Simplyfy nav_status
    nav_status_adapter = {
        1: "Stationary", 5: "Stationary", 6: "Stationary",
        0: "Underway", 8: "Underway", 9: "Underway", 10: "Underway",
        3: "Privileged", 4: "Privileged", 7: "Privileged", 11: "Privileged", 12: "Privileged",
        2: "Abnormal",
        13: "Unknown", 15: "Unknown"
    }
    df['nav_status_simple'] = df['nav_status'].map(nav_status_adapter)
    
    try:
        # Categorical features

        categorical_features = ['nav_status_simple', 'vessel_type_simple']

        for categorical_feature_ in categorical_features:
            dummies = pd.get_dummies(df[categorical_feature_], columns=[categorical_feature_])

            for cluster_size in [2, 3, 5]:
                cluster_label_col = f'cluster_label_{cluster_size}'
                dummies_ = dummies.copy()
                dummies_[cluster_label_col] = df[cluster_label_col]
                dummies_ = dummies_[dummies_[cluster_label_col] != -1] if not args.include_noise else dummies_
                

                # 4. Calculate the "Global Mean" (The expected % of each category)
                global_means = dummies_.drop(columns=[cluster_label_col]).mean()

                # 5. Calculate the "Cluster Mean" (The actual % in each cluster)
                cluster_means = dummies_.groupby(cluster_label_col).mean()

                # 6. Calculate Enrichment (The Difference)
                # This subtracts the global mean from every cluster row
                enrichment_scores = cluster_means.sub(global_means, axis=1)
                
                pickle_and_artifact(enrichment_scores, os.path.join(args.output_dir, f'categorical_enrichment_{categorical_feature_}_cure_{cluster_size}_clusters.pkl'), logger)

                # -------------------------------------------------------
                # PLOTTING
                # -------------------------------------------------------
                plt.figure(figsize=(12, 8))

                sns.heatmap(
                    enrichment_scores.T,  # Transpose so Features are rows, Clusters are columns
                    annot=True,
                    cmap='RdBu_r',        # Red = Over-represented, Blue = Under-represented
                    center=0,             # White = Expected baseline
                    fmt='.2f',
                    vmin=-0.5,            # Cap colors at +/- 50% shift to keep contrast
                    vmax=0.5
                )

                plt.title(f"Categorical Feature Enrichment\n(Difference from Global Average) - CURE with {cluster_size} Clusters")
                plt.xlabel("Cluster Label")
                category = categorical_feature_.replace('_', ' ').replace(' simple', '').title()
                plt.ylabel(category)
                plt.tight_layout()
                save_plot_and_artifact(plt, os.path.join(args.output_dir, 'plots', f'categorical_enrichment_{categorical_feature_}_cure_{cluster_size}_clusters.png'), logger)
                plt.close()
                logger.info(f"Saved Categorical Enrichment plot for {categorical_feature_} CURE {cluster_size} clusters to {os.path.join(args.output_dir, 'plots', f'categorical_enrichment_{categorical_feature_}_cure_{cluster_size}_clusters.png')}")
    except Exception as e:
        logger.warning(f"Categorical feature enrichment plotting failed: {e}")
                
    try:
        for cluster_size in [2, 3, 5]:
            cluster_label_col = f'cluster_label_{cluster_size}'
            
            plot_df = df.copy()
            plot_df = plot_df[[cluster_label_col, 'hour_start', 'month']]
            plot_df.columns = ['Cluster', 'Hour', 'Month']

            # Filter out noise (-1) if you only care about valid clusters
            plot_df = plot_df[plot_df['Cluster'] != -1] if not args.include_noise else plot_df

            # 2. Create a Crosstab (Contingency Table)
            # normalize='index' ensures rows sum to 1 (shows % of cluster occurring at that hour)
            ct_hour = pd.crosstab(plot_df['Cluster'], plot_df['Hour'], normalize='index')
            ct_month = pd.crosstab(plot_df['Cluster'], plot_df['Month'], normalize='index')

            pickle_and_artifact(ct_hour, os.path.join(args.output_dir, f'crosstab_hour_cure_{cluster_size}_clusters.pkl'), logger)
            pickle_and_artifact(ct_month, os.path.join(args.output_dir, f'crosstab_month_cure_{cluster_size}_clusters.pkl'), logger)
            
            # 3. Plot
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))

            sns.heatmap(ct_hour, cmap="viridis", annot=False, ax=axes[0])
            axes[0].set_title(f"Distribution of Clusters by Hour (Normalized)")

            sns.heatmap(ct_month, cmap="viridis", annot=False, ax=axes[1])
            axes[1].set_title(f"Distribution of Clusters by Month (Normalized)")
            
            plt.suptitle(f"CURE Clustering with {cluster_size} Clusters")

            save_plot_and_artifact(fig, os.path.join(args.output_dir, 'plots', f'crosstab_time_cure_{cluster_size}_clusters.png'), logger)
            plt.close(fig)
            logger.info(f"Saved Crosstab Time plot for CURE {cluster_size} clusters to {os.path.join(args.output_dir, 'plots', f'crosstab_time_cure_{cluster_size}_clusters.png')}")
    except Exception as e:
        logger.warning(f"Crosstab time plotting failed: {e}")
    
    logger.info("Clustering evaluation completed.")
    logger.finish()
    