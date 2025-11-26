import numpy as np
import time
import tracemalloc
import matplotlib.pyplot as plt
from src.clustering.hierarchical import CURE

def generate_synthetic_data(n_samples=1000, n_features=256):
    """Generates random synthetic data."""
    return np.random.rand(n_samples, n_features) * 100

def benchmark_clustering(X, step=100):
    """
    Runs the benchmark on subsets of X for the CURE algorithm only.
    """
    L = len(X)
    sample_sizes = range(step, L + 1, step)
    
    # Store results for CURE
    results = {
        'cure': {'time': [], 'memory': []}
    }
    
    print(f"{'N Samples':<10} | {'CURE Time':<10} | {'CURE Mem (MB)':<15}")
    print("-" * 45)

    for n in sample_sizes:
        subset = X[:n]
        
        # --- Benchmark CURE ---
        # For CURE, fit() is constant time (clustering sample), 
        # but get_labels() is O(N) (assigning full dataset).
        tracemalloc.start()
        start_time = time.time()
        
        # Using a fixed sample size
        cure = CURE(sample_size=150) 
        cure.fit(subset)
        # We must call get_labels to trigger the assignment of the N points
        cure.get_labels(n_clusters=3)
        
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results['cure']['time'].append(end_time - start_time)
        results['cure']['memory'].append(peak / (1024 * 1024))
        
        # Print progress
        print(f"{n:<10} | {results['cure']['time'][-1]:<10.4f} | {results['cure']['memory'][-1]:<10.2f}")

    return sample_sizes, results

def plot_results(sample_sizes, results):
    """
    Plots performance metrics for CURE.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Time
    ax1.plot(sample_sizes, results['cure']['time'], 's-', label='CURE', color='green')
    ax1.set_xlabel('Number of Samples (N)')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('CURE Execution Time')
    ax1.legend()
    ax1.grid(True)

    # Plot Memory
    ax2.plot(sample_sizes, results['cure']['memory'], 's--', label='CURE', color='green')
    ax2.set_xlabel('Number of Samples (N)')
    ax2.set_ylabel('Peak Memory (MB)')
    ax2.set_title('CURE Memory Usage')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Since CURE is O(N), we can test with significantly more samples than before
    TOTAL_SAMPLES = 5000 
    STEP_SIZE = 500
    
    print(f"Generating {TOTAL_SAMPLES} samples...")
    X = generate_synthetic_data(n_samples=TOTAL_SAMPLES)
    
    sizes, results = benchmark_clustering(X, step=STEP_SIZE)
    plot_results(sizes, results)