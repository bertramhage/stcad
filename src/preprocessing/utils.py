"""
This script is a collection of utility functions for preprocessing AIS data. Source: [CIA-Oceanix/GeoTrackNet](https://github.com/CIA-Oceanix/GeoTrackNet)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import scipy.ndimage as ndimage
import sys
sys.path.append('..')
sys.path.append('Data')
from pyproj import Geod
geod = Geod(ellps='WGS84')

from scipy import sparse

AVG_EARTH_RADIUS = 6378.137  # in km
SPEED_MAX = 30 # knot
FIG_DPI = 150

LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))
# Your original function, optimized for sparse matrices
def trackOutlier(A_csr: sparse.csr_matrix):
    """
    Koyak algorithm to perform outlier identification,
    optimized to use a sparse CSR matrix.
    
    INPUT:
        A_csr   : (nxn) symmetric CSR sparse matrix of anomaly indicators
    OUTPUT:
        o       : n-vector outlier indicators
    """
    n = A_csr.shape[0]
    
    # Calculate row sums. This is fast and returns a (n, 1) matrix.
    # .flatten() converts it to a 1D array.
    b = np.asarray(A_csr.sum(axis=1)).flatten()
    o = np.zeros(n, dtype=bool)
    
    while(np.max(b) > 0):
        r = np.argmax(b) # Find row with max anomalies (still O(N))
        o[r] = 1         # Mark as outlier
        b[r] = 0         # Remove from consideration
        
        # --- OPTIMIZED INNER LOOP ---
        # Get all non-zero column indices for row r.
        # This is extremely fast and gives only the ~8 neighbors.
        neighbors = A_csr.indices[A_csr.indptr[r]:A_csr.indptr[r+1]]
        
        # This loop now runs ~8 times, not N times
        for j in neighbors:
            if not o[j]: # If the neighbor is not already an outlier
                b[j] -= 1 # Decrement its anomaly count
                
    return o
    
#===============================================================================
#===============================================================================

# Your original function, optimized to *create* a sparse matrix
def detectOutlier(track, speed_max):
    """
    Removes anomalous AIS messages from an AIS track.
    Optimized to use sparse matrices to avoid O(N^2) memory usage.
    
    INPUT:
        track       : (nxd) matrix: [Timestamp, Lat, Lon, Speed]
        speed_max   : knot
    OUTPUT:
        o_report    : n-vector outlier indicators (from reported speed)
        o_calcul    : (n_filtered)-vector outlier indicators (from calculated speed)
    """
    # Remove anomalous reported speed
    o_report = track[:, 3] > speed_max  # Speed in track is in knot
    if o_report.all():
        # All points are outliers, return empty calculated outliers
        return o_report, np.array([], dtype=bool)
        
    track_filtered = track[np.invert(o_report)]
    
    # Handle edge case where filtering removes everything
    if track_filtered.shape[0] < 2:
        return o_report, np.array([], dtype=bool)

    N = track_filtered.shape[0]
    
    # --- MEMORY FIX ---
    # Create a List-in-List sparse matrix, best for incremental building.
    # Use int8 for minimal memory.
    A = sparse.lil_matrix((N, N), dtype=np.int8)
    
    # Anomalous calculated-speed
    # 0.514444 is knots to m/s
    speed_max_ms = speed_max * 0.514444 
    
    for i in range(1, 5): # the i-th diagonal
        if N <= i: # Stop if track is shorter than the offset
            break
            
        # Geod.inv calculates distance (d)
        _, _, d = geod.inv(track_filtered[:-i, 2], track_filtered[:-i, 1],
                           track_filtered[i:, 2],  track_filtered[i:, 1])
                           
        delta_t = track_filtered[i:, 0] - track_filtered[:-i, 0]
        
        # Create a mask for valid time deltas to avoid divide-by-zero
        valid_time = delta_t > 2 # Original code had 2s threshold
        
        # Initialize speed_ratio with zeros
        speed_ratio = np.zeros_like(d)
        
        # Calculate speed only where delta_t is valid
        np.divide(d, delta_t, out=speed_ratio, where=valid_time)
        
        # Find where speed is too high AND time was valid
        cond = (speed_ratio > speed_max_ms) & valid_time
        
        abnormal_idx = np.nonzero(cond)[0]
        
        if abnormal_idx.size > 0:
            A[abnormal_idx, abnormal_idx + i] = 1
            A[abnormal_idx + i, abnormal_idx] = 1    

    # --- TIME FIX ---
    # Convert to Compressed Sparse Row (CSR) format
    # This is *much* faster for the row summing and lookups in trackOutlier
    A_csr = A.tocsr()
    
    o_calcul = trackOutlier(A_csr)
            
    return o_report, o_calcul
    
def interpolate(t, track):
    """
    Interpolating the AIS message of vessel at a specific "t".
    INPUT:
        - t : 
        - track     : AIS track, whose structure is
                     [LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI]
    OUTPUT:
        - [LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI]
                        
    """
    
    before_p = np.nonzero(t >= track[:,TIMESTAMP])[0]
    after_p = np.nonzero(t < track[:,TIMESTAMP])[0]
   
    if (len(before_p) > 0) and (len(after_p) > 0):
        apos = after_p[0]
        bpos = before_p[-1]    
        # Interpolation
        dt_full = float(track[apos,TIMESTAMP] - track[bpos,TIMESTAMP])
        if (abs(dt_full) > 2*3600):
            return None
        dt_interp = float(t - track[bpos,TIMESTAMP])
        try:
            az, _, dist = geod.inv(track[bpos,LON],
                                   track[bpos,LAT],
                                   track[apos,LON],
                                   track[apos,LAT])
            dist_interp = dist*(dt_interp/dt_full)
            lon_interp, lat_interp, _ = geod.fwd(track[bpos,LON], track[bpos,LAT],
                                               az, dist_interp)
            speed_interp = (track[apos,SOG] - track[bpos,SOG])*(dt_interp/dt_full) + track[bpos,SOG]
            course_interp = (track[apos,COG] - track[bpos,COG] )*(dt_interp/dt_full) + track[bpos,COG]
            heading_interp = (track[apos,HEADING] - track[bpos,HEADING])*(dt_interp/dt_full) + track[bpos,HEADING]  
            rot_interp = (track[apos,ROT] - track[bpos,ROT])*(dt_interp/dt_full) + track[bpos,ROT]
            if dt_interp > (dt_full/2):
                nav_interp = track[apos,NAV_STT]
            else:
                nav_interp = track[bpos,NAV_STT]                             
        except:
            return None
        return np.array([lat_interp, lon_interp,
                         speed_interp, course_interp, 
                         heading_interp, rot_interp, 
                         nav_interp,t,
                         track[0,MMSI]])
    else:
        return None