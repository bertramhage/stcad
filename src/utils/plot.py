# Various plotting helpers

import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
from src.preprocessing.preprocessing import de_normalize_track
from src.preprocessing.preprocessing import (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
                                             LAT, LON) # Indeces

def plot_scatter(data: np.ndarray, title: str = None, xlab: str = None, ylab: str = None, color: str = 'blue'):
    plt.figure(figsize=(10, 8))
    _ = plt.scatter(data[:, 0], data[:, 1], c=color, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True)
    plt.show()

def plot_trajectories(track_list: list[dict], title: str | None = None, show_plot: bool = True):
    """
    Plots multiple trajectories on the same plot.
    
    Expects a list of dictionaries, each containing:
    - 'track': np.ndarray of shape (N, 4) with trajectory data
    - 'color': str, color for the trajectory line
    - 'label': str, optional label for the trajectory (e.g. cluster label)
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    all_lats = [] # For mean latitude calculation
    seen_labels = set()
    show_legend = False

    for idx, track_dict in enumerate(track_list):
        tracks = track_dict['track']
        color = track_dict['color']
        
        if 'label' in track_dict:
            label = track_dict['label']
            show_legend = True
            if label in seen_labels:
                label = "_nolegend_"
            else:
                seen_labels.add(label)
        else:
            label = f'Track {idx + 1}'

        if tracks[0, LAT] < 1: # Assume normalized
            tracks = de_normalize_track(tracks)
        
        lats = tracks[:, LAT]
        lons = tracks[:, LON]
        all_lats.extend(lats)
        
        ax.plot(lons, lats, color=color, linewidth=2, 
                alpha=0.7, label=label, zorder=2)
        
        # Mark start and end points
        ax.plot(lons[0], lats[0], 'o', color=color, markersize=10, 
                markerfacecolor='none', markeredgewidth=2, zorder=3)
        ax.plot(lons[-1], lats[-1], 's', color=color, markersize=10, 
                markerfacecolor='none', markeredgewidth=2, zorder=3)
    
    if show_legend:
        ax.legend()

    mean_lat = np.mean(all_lats)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Earth is round and thus the aspect ratio needs correction
    # we use the mean latitude of all tracks for this
    if np.cos(np.deg2rad(mean_lat)) > 0:
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
    else:
        ax.set_aspect('equal')
    
    try:
        cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.CartoDB.Positron, zorder=1)
    except Exception as e:
        print(f"Warning: Could not add basemap due to: {e}")
    
    if title:
        plt.suptitle(title, fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    if show_plot:
        plt.show()
        return None
    else:
        return fig
    
def plot_lollipop(ax, data, title):
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in data['Score']]
    
    ax.hlines(y=data['Feature'], xmin=0, xmax=data['Score'], 
              color=colors, alpha=0.4, linewidth=2)
    
    ax.scatter(x=data['Score'], y=data['Feature'], 
               color=colors, s=80, alpha=1, zorder=3)
    
    ax.axvline(x=0, color='grey', alpha=0.5, linestyle='--', linewidth=1)
    ax.grid(axis='x', color='grey', alpha=0.2)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(left=-1.7, right=1.2)
    
    ax.set_title(title, x=-0.23, y=1.0, loc='left', fontsize=16, fontweight='bold', color='#000', pad=20, fontfamily='times new roman')
