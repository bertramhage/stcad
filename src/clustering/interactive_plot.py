import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # Re-using this for color generation
from scipy.cluster.hierarchy import dendrogram

def plot_interactive_dendrogram(fitted_model, save_path="dendrogram.html", color_by_size=True, **kwargs):
    """
    Creates an interactive HTML dendrogram using Plotly.
    """
    if fitted_model.agg_.linkage_matrix_ is None:
        print("Error: Model is not fit yet.")
        return

    # --- 1. Setup Color Logic (Same as before) ---
    link_color_func = None
    if color_by_size:
        n_samples = len(fitted_model.agg_.linkage_matrix_) + 1
        cluster_sizes = {i: 1 for i in range(n_samples)}
        for i, row in enumerate(fitted_model.agg_.linkage_matrix_):
            cluster_sizes[n_samples + i] = int(row[3])

        cmap = plt.get_cmap('viridis')
        norm = mcolors.LogNorm(vmin=1, vmax=n_samples)

        def size_color_func(k):
            return mcolors.to_hex(cmap(norm(cluster_sizes.get(k, 1))))
        
        link_color_func = size_color_func

    # --- 2. Get Dendrogram Data from Scipy (no_plot=True) ---
    # This calculates the X and Y coordinates for us
    dendro_data = dendrogram(
        fitted_model.agg_.linkage_matrix_,
        link_color_func=link_color_func,
        no_plot=True, # <--- Key: Don't draw, just give me data
        **kwargs
    )

    # --- 3. Build Plotly Figure ---
    fig = go.Figure()

    # The dendrogram returns lists of coordinates for the 'U' shapes
    # icoord = x-coords, dcoord = y-coords
    # color_list = color for each link
    
    for i, (xs, ys) in enumerate(zip(dendro_data['icoord'], dendro_data['dcoord'])):
        color = dendro_data['color_list'][i]
        
        # Plotly logic: drawn generic shapes for the tree
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='lines',
            line=dict(color=color, width=2),
            hoverinfo='y', # Show distance on hover
            showlegend=False
        ))

    # --- 4. Styling ---
    fig.update_layout(
        title="Interactive Dendrogram",
        xaxis_title="Sample Index",
        yaxis_title="Distance",
        width=1000,
        height=600,
        hovermode='closest',
        plot_bgcolor='white'
    )
    
    # Add a dummy marker for the Colorbar (Plotly hack)
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

    # --- 5. Save ---
    print(f"Saving interactive plot to {save_path}...")
    fig.write_html(save_path)