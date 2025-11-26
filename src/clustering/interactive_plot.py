import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import dendrogram
import numpy as np

def plot_interactive_dendrogram(fitted_model, save_path="dendrogram.html", color_by_size=True, **kwargs):
    """
    Creates an interactive HTML dendrogram using Plotly, handling Ghost Merges (Noise).
    """
    if fitted_model.agg_ is None or fitted_model.agg_.linkage_matrix_ is None:
        print("Error: Model is not fit yet.")
        return

    linkage_matrix = fitted_model.agg_.linkage_matrix_
    n_samples = len(linkage_matrix) + 1

    # --- 1. Determine Ghost Threshold ---
    # If we pruned points, the max distance is likely the ghost distance.
    # We set the threshold just below the max to catch those artificial merges.
    distances = linkage_matrix[:, 2]
    max_dist = np.max(distances)
    
    # Check if pruning actually happened
    has_pruned = len(getattr(fitted_model.agg_, 'pruned_indices_', [])) > 0
    ghost_threshold = max_dist * 0.99 if has_pruned else np.inf

    # --- 2. Setup Color Logic ---
    link_color_func = None
    if color_by_size:
        cluster_sizes = {i: 1 for i in range(n_samples)}
        for i, row in enumerate(linkage_matrix):
            cluster_sizes[n_samples + i] = int(row[3])

        cmap = plt.get_cmap('viridis')
        norm = mcolors.LogNorm(vmin=1, vmax=n_samples)

        def size_color_func(k):
            # A. Check for Ghost Merge (Noise)
            # Linkage rows are indexed 0 to N-2.
            # Cluster IDs for merges are N to 2N-2.
            if k >= n_samples:
                idx = k - n_samples
                # If this link's distance is super high, it's a ghost merge -> Gray
                if linkage_matrix[idx, 2] >= ghost_threshold:
                    return '#d3d3d3' # Light Gray

            # B. Standard Size Coloring
            return mcolors.to_hex(cmap(norm(cluster_sizes.get(k, 1))))
        
        link_color_func = size_color_func

    # --- 3. Get Dendrogram Data ---
    dendro_data = dendrogram(
        linkage_matrix,
        link_color_func=link_color_func,
        no_plot=True, 
        **kwargs
    )

    # --- 4. Build Plotly Figure ---
    fig = go.Figure()

    for i, (xs, ys) in enumerate(zip(dendro_data['icoord'], dendro_data['dcoord'])):
        color = dendro_data['color_list'][i]
        
        # Add hover text to show if it is noise
        # Note: We can infer if it's noise by the color we just assigned
        hover_txt = f"Distance: {max(ys):.3f}"
        if color == '#d3d3d3':
             hover_txt += "<br>Status: PRUNED NOISE"

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='lines',
            line=dict(color=color, width=2),
            text=hover_txt,
            hoverinfo='text',
            showlegend=False
        ))

    # --- 5. Styling ---
    fig.update_layout(
        title="Interactive Dendrogram (Gray = Pruned Noise)",
        xaxis_title="Sample Index",
        yaxis_title="Distance",
        width=1000,
        height=600,
        hovermode='closest',
        plot_bgcolor='white'
    )
    
    # Add Colorbar (Dummy trace)
    if color_by_size:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale='Viridis',
                cmin=1,
                cmax=n_samples,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Cluster Size (Log)", side="right")
                )
            ),
            showlegend=False
        ))

    print(f"Saving interactive plot to {save_path}...")
    fig.write_html(save_path)