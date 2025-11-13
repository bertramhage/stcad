# Adapted from [CIA-Oceanix/GeoTrackNet](https://github.com/CIA-Oceanix/GeoTrackNet)

import numpy as np
from src.preprocessing import utils
import copy

LON_MIN = 5.0
LON_MAX = 17.0
LAT_MIN = 54.0
LAT_MAX = 59.0

LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30.0  # knots
DURATION_MAX = 24 #h

LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))

def preprocess_mmsi_track(track_data: np.ndarray) -> dict:
    
    # Prepare results dictionary
    preprocessing_results = {"num_messages": track_data.shape[0]}
   
    ## FILTER
    # We filter out messages that are outside the defined boundaries or have abnormal speeds.
    
    # Boundary
    lat_idx = np.logical_or((track_data[:,LAT] > LAT_MAX),
                            (track_data[:,LAT] < LAT_MIN))
    track_data = track_data[np.logical_not(lat_idx)]
    lon_idx = np.logical_or((track_data[:,LON] > LON_MAX),
                            (track_data[:,LON] < LON_MIN))
    track_data = track_data[np.logical_not(lon_idx)]

    # Abnormal speeds
    abnormal_speed_idx = track_data[:,SOG] > SPEED_MAX
    track_data = track_data[np.logical_not(abnormal_speed_idx)]
    
    preprocessing_results["num_discarded_filtered"] = preprocessing_results["num_messages"] - track_data.shape[0]
            
    ## VOYAGES SPLITTING 
    # Split the track into voyages based on time gaps > 2 hours
    
    count = 0
    voyages = dict()
    INTERVAL_MAX = 2*3600 # 2h

    v = track_data
    # Intervals between successive messages in a track
    intervals = v[1:,TIMESTAMP] - v[:-1,TIMESTAMP]
    idx = np.where(intervals > INTERVAL_MAX)[0]
    if len(idx) == 0:
        voyages[count] = v
        count += 1
    else:
        tmp = np.split(v,idx+1)
        for t in tmp:
            voyages[count] = t
            count += 1
    
    preprocessing_results["num_initial_voyages"] = len(voyages)
            
    # REMOVING SHORT VOYAGES
    # Remove voyages with less than 20 messages or duration < 4 hours
    for k in list(voyages.keys()):
        duration = voyages[k][-1,TIMESTAMP] - voyages[k][0,TIMESTAMP]
        if (len(voyages[k]) < 20) or (duration < 4*3600):
            voyages.pop(k, None)
    
    preprocessing_results["num_voyages_after_duration_filter"] = len(voyages)
    
    # REMOVING OUTLIERS
    # An AIS message is considered as beging anomalous if the speed is infeasible (> speed_max)
    error_count = 0
    for k in list(voyages.keys()):
        track = voyages[k][:,[TIMESTAMP,LAT,LON,SOG]] # [Timestamp, Lat, Lon, Speed]
        try:
            o_report, o_calcul = utils.detectOutlier(track, speed_max = SPEED_MAX)
            if o_report.all() or o_calcul.all():
                voyages.pop(k, None)
            else:
                voyages[k] = voyages[k][np.invert(o_report)]
                voyages[k] = voyages[k][np.invert(o_calcul)]
        except Exception as e:
            error_count += 1
            voyages.pop(k, None)
    
    preprocessing_results["num_voyages_after_outlier_removal"] = len(voyages)
    preprocessing_results["num_outlier_removal_errors"] = error_count
            
    ## SAMPLING
    # Down sample each voyage to 5 min interval using interpolation
    Vs = dict()
    count = 0
    error_count = 0
    for k in list(voyages.keys()):
        v = voyages[k]
        sampling_track = np.empty((0, 9))
        for t in range(int(v[0,TIMESTAMP]), int(v[-1,TIMESTAMP]), 300): # 5 min
            tmp = utils.interpolate(t,v)
            if tmp is not None:
                sampling_track = np.vstack([sampling_track, tmp])
            else:
                sampling_track = None
                error_count += 1
                break
        if sampling_track is not None:
            Vs[count] = sampling_track
            count += 1
    
    del voyages  # Free memory
            
    preprocessing_results["num_voyages_after_sampling"] = len(Vs)
    preprocessing_results["num_sampling_errors"] = error_count
            
    ## RE-SPLITTING
    Data = dict()
    count = 0
    for k in list(Vs.keys()): 
        v = Vs[k]
        # Split AIS track into small tracks whose duration <= 1 day
        idx = np.arange(0, len(v), 12*DURATION_MAX)[1:]
        tmp = np.split(v,idx)
        for subtrack in tmp:
            # only use tracks whose duration >= 4 hours
            if len(subtrack) >= 12*4:
                Data[count] = subtrack
                count += 1
                
    ## REMOVING LOW SPEED TRACKS
    for k in list(Data.keys()):
        d_L = float(len(Data[k]))
        if np.count_nonzero(Data[k][:,SOG] < 2)/d_L > 0.8:
            Data.pop(k,None)
    
    preprocessing_results["num_final_voyages"] = len(Data)

    ## NORMALISATION
    for k in list(Data.keys()):
        v = Data[k]
        v[:,LAT] = (v[:,LAT] - LAT_MIN)/(LAT_MAX-LAT_MIN)
        v[:,LON] = (v[:,LON] - LON_MIN)/(LON_MAX-LON_MIN)
        v[:,SOG][v[:,SOG] > SPEED_MAX] = SPEED_MAX
        v[:,SOG] = v[:,SOG]/SPEED_MAX
        v[:,COG] = v[:,COG]/360.0
    
    return Data, preprocessing_results

def de_normalize_track(track: np.ndarray) -> np.ndarray:
    """Denormalizes a single track."""
    denorm_track = copy.deepcopy(track)

    denorm_track[:, LAT] = denorm_track[:, LAT] * (LAT_MAX - LAT_MIN) + LAT_MIN
    denorm_track[:, LON] = denorm_track[:, LON] * (LON_MAX - LON_MIN) + LON_MIN
    denorm_track[:, SOG] = denorm_track[:, SOG] * SPEED_MAX
    denorm_track[:, COG] = denorm_track[:, COG] * 360.0
    
    return denorm_track