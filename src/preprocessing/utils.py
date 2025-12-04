"""
This script is a collection of utility functions for preprocessing AIS data.
Original source is: [CIA-Oceanix/GeoTrackNet](https://github.com/CIA-Oceanix/GeoTrackNet)
although the script has been modified to use a sparse matrix, improving memory efficiency
"""

import numpy as np
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

def trackOutlier(A_csr: sparse.csr_matrix):
    """
    Koyak algorithm to perform outlier identification,
    optimized to use a sparse CSR matrix.
    
    INPUT:
        A_csr: (nxn) symmetric CSR sparse matrix of anomaly indicators
    OUTPUT:
        o: n-vector outlier indicators
    """
    n = A_csr.shape[0]
    
    b = np.asarray(A_csr.sum(axis=1)).flatten()
    o = np.zeros(n, dtype=bool)
    
    while(np.max(b) > 0):
        r = np.argmax(b) # Find row with max anomalies
        o[r] = 1 # Mark as outlier
        b[r] = 0
        
        # Get all non-zero column indices for row r.
        neighbors = A_csr.indices[A_csr.indptr[r]:A_csr.indptr[r+1]]
        
        for j in neighbors:
            if not o[j]: # If the neighbor is not already an outlier
                b[j] -= 1 # Decrement its anomaly count
                
    return o

def detectOutlier(track, speed_max):
    """
    Removes anomalous AIS messages from an AIS track.
    
    INPUT:
        track: (nxd) matrix: [Timestamp, Lat, Lon, Speed]
        speed_max: knot
    OUTPUT:
        o_report: n-vector outlier indicators (from reported speed)
        o_calcul: (n_filtered)-vector outlier indicators (from calculated speed)
    """
    # Remove anomalous reported speed
    o_report = track[:, 3] > speed_max
    if o_report.all():
        # All points are outliers
        return o_report, np.array([], dtype=bool)
        
    track_filtered = track[np.invert(o_report)]
    
    # If filtering removes everything
    if track_filtered.shape[0] < 2:
        return o_report, np.array([], dtype=bool)

    N = track_filtered.shape[0]
    
    # Use a sparse matrix, otherwise memory usage scales O(N^2)
    A = sparse.lil_matrix((N, N), dtype=np.int8)
    
    speed_max_ms = speed_max * 0.514444 # knots to m/s
    
    for i in range(1, 5):
        if N <= i:
            break
            
        # calculate distance
        _, _, d = geod.inv(track_filtered[:-i, 2], track_filtered[:-i, 1],
                           track_filtered[i:, 2],  track_filtered[i:, 1])
                           
        delta_t = track_filtered[i:, 0] - track_filtered[:-i, 0]
        
        valid_time = delta_t > 2
        speed_ratio = np.zeros_like(d)
        np.divide(d, delta_t, out=speed_ratio, where=valid_time)
        cond = (speed_ratio > speed_max_ms) & valid_time
        abnormal_idx = np.nonzero(cond)[0]
        
        if abnormal_idx.size > 0:
            A[abnormal_idx, abnormal_idx + i] = 1
            A[abnormal_idx + i, abnormal_idx] = 1    

    # convert to CSR format for efficient slicing
    A_csr = A.tocsr()
    
    o_calcul = trackOutlier(A_csr)
            
    return o_report, o_calcul
    
def interpolate(t, track):
    """
    Interpolating the AIS message of vessel at a specific "t".
    INPUT:
        - t: timestamp to interpolate
        - track: AIS track, whose structure is [LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI]
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