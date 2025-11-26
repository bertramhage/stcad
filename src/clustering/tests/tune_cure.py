import numpy as np
import time
from src.clustering.hierarchical import CURE
from src.clustering.utils import davies_bouldin_index
from argparse import ArgumentParser
from src.utils.datasets import AISDataset
from src.sequence_modelling.models import AISBERT
from src.sequence_modelling.utils import get_embeddings
from torch.utils.data import Subset
from src.utils.logging import CustomLogger

def grid_search_cure(X_train, n_clusters_target=4, logger=None):
    """
    Performs grid search on CURE hyperparameters.
    """
    # STRATEGY UPDATE:
    # 1. Use a small, fixed sample_size for tuning reps and compression.
    #    (sample_size=300 is roughly 150s unoptimized, much faster if optimized)
    # 2. We assume the best 'shape parameters' (reps/compression) found on the 
    #    small sample will hold true for the large sample.
    
    param_grid = {
        'sample_size': [75, 150], # Keep this low for the search phase
        'n_representatives': [5, 10, 20],
        'compression': [0.2, 0.5] 
    }
    
    logger.log_config(param_grid)
    
    best_score = np.inf
    best_params = {}
    
    logger.info(f"{'Sample':<8} | {'Reps':<5} | {'Comp.':<6} | {'DB Index (Lower is better)':<25} | {'Time (s)':<10}")
    logger.info("-" * 65)

    i = 1
    for sample_size in param_grid['sample_size']:
        for n_reps in param_grid['n_representatives']:
            for comp in param_grid['compression']:
                
                start_time = time.time()
                
                # Initialize CURE
                cure = CURE(sample_size=sample_size, 
                            n_representatives=n_reps, 
                            compression=comp,
                            linkage='single')
                
                # Fit on training data
                cure.fit(X_train)
                
                # Calculate scores for multiple cluster numbers
                if not isinstance(n_clusters_target, list):
                    n_clusters_target = [n_clusters_target]
                
                scores = []
                for n_clusters in n_clusters_target:
                    labels = cure.get_labels(n_clusters=n_clusters)
                    score = davies_bouldin_index(X_train, labels)
                    scores.append(score)
                
                # Use average score across all cluster numbers
                avg_score = np.mean(scores)
                min_score = np.min(scores)
                
                elapsed = time.time() - start_time
                
                score_str = f"{min_score:.4f} (min of {n_clusters_target}: {', '.join(f'{s:.4f}' for s in scores)})"
                logger.info(f"{sample_size:<8} | {n_reps:<5} | {comp:<6.1f} | {score_str:<25} | {elapsed:<10.2f}")
                
                if min_score < best_score:
                    best_score = min_score
                    best_params = {
                        'sample_size': sample_size,
                        'n_representatives': n_reps,
                        'compression': comp
                    }
                
                logger.log_metrics({'best_score': best_score}, step=i)
                i += 1

    logger.info("-" * 65)
    logger.info(f"Best Parameters (found on small sample): {best_params}")
    logger.info(f"Best DB Index: {best_score:.4f}")
    return best_params

if __name__ == "__main__":
    parser = ArgumentParser(description="CURE Hyperparameter Tuning")
    parser.add_argument('--data_path', type=str, default=None, help="Path to dataset (if None, synthetic data is used)")
    parser.add_argument('--sample_size', type=int, default=1000, help="Number of samples to use for tuning")
    args = parser.parse_args()
    
    logger = CustomLogger(project_name='Computational-Tools', group='clustering', run_name='tune_cure', use_wandb=True)
    
    model = AISBERT.from_pretrained('models/pretrained_models/pretrained_bert-3026')
    ds = AISDataset(data_dir=args.data_path)
    
    all_indices = np.arange(len(ds))
    picked_indices = np.random.choice(all_indices, size=args.sample_size, replace=False)
    ds_subset = Subset(ds, picked_indices)
    
    embeddings, _ = get_embeddings(ds_subset, model, l2_normalize=True)
    logger.info(f"Extracted embeddings shape: {embeddings.shape}")
    
    # Run Grid Search
    logger.info("\n--- Starting Grid Search (Small Sample) ---")
    best_params = grid_search_cure(embeddings, n_clusters_target=[2,5, 10], logger=logger)
    
    logger.info("\n\n--- Recommendations for Full 500k Run ---")
    logger.info("Use the 'shape' parameters found above, but increase sample_size for better skeleton:")
    logger.info(f"1. Initialize: cure = CURE(sample_size={best_params['sample_size']}, "
          f"n_representatives={best_params['n_representatives']}, "
          f"compression={best_params['compression']})")
    logger.info("2. Call: cure.fit(X_full_500k)")
    logger.info("3. Loop: labels_k = cure.get_labels(k) for k in [2, 5, 10...]")
    
    logger.log_summary({
        'best_n_representatives': best_params['n_representatives'],
        'best_compression': best_params['compression'],
        'best_sample_size': best_params['sample_size'],
    })
    
    logger.finish()