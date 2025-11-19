# For plotting trajectories

import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
from src.preprocessing.preprocessing import de_normalize_track
from src.preprocessing.preprocessing import (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
                                             LAT, LON) # Indeces

def plot_trajectories(track_list: list[dict]):
    """
    Plots multiple trajectories on the same plot.
    
    Expects a list of dictionaries, each containing:
    - 'track': np.ndarray of shape (N, 4) with trajectory data
    - 'color': str, color for the trajectory line
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    all_lats = [] # For mean latitude calculation
    for idx, track_dict in enumerate(track_list):
        tracks = track_dict['track']
        color = track_dict['color']
        
        if tracks[0, LAT] < 1: # Assume normalized
            tracks = de_normalize_track(tracks)
        
        lats = tracks[:, LAT]
        lons = tracks[:, LON]
        all_lats.extend(lats)
        
        ax.plot(lons, lats, color=color, linewidth=2, 
                alpha=0.7, label=f'Track {idx + 1}', zorder=2)
        
        # Mark start and end points
        ax.plot(lons[0], lats[0], 'o', color=color, markersize=10, 
                markerfacecolor='none', markeredgewidth=2, zorder=3)
        ax.plot(lons[-1], lats[-1], 's', color=color, markersize=10, 
                markerfacecolor='none', markeredgewidth=2, zorder=3)
    
    mean_lat = np.mean(all_lats)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Vessel Trajectories', fontsize=16, fontweight='bold')
    
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Earth is round and thus the aspect ratio needs correction
    # we use the mean latitude of all tracks for this
    if np.cos(np.deg2rad(mean_lat)) > 0:
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
    else:
        ax.set_aspect('equal')
    
    cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.CartoDB.Positron, zorder=1)
    
    plt.tight_layout()
    plt.show()