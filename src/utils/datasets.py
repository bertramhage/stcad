"""Customized Pytorch Dataset."""

import numpy as np
import os
import joblib
import torch
from torch.utils.data import Dataset
from datetime import datetime
from geopy.distance import geodesic
from src.preprocessing.preprocessing import de_normalize_track

class AISDataset(Dataset):
    """Customized Pytorch dataset that loads from disk.
    """
    def __init__(self, 
                 data_dir,
                 file_extension=".pkl"):
        """
        Args
            data_dir: path to directory containing data files
            max_seqlen: max sequence length
            file_extension: file extension to look for
        """    
            
        self.data_dir = data_dir
        
        # Build list of filenames
        self.file_list = [
            f for f in os.listdir(data_dir) 
            if f.endswith(file_extension) and 
            (not f == 'vessel_types.pkl')
        ]
        self.file_list.sort()

    def __len__(self):
        return len(self.file_list)
    
    def _load_file(self, filepath):
        """Load a single data file. Modify based on your file format."""
        V = joblib.load(filepath)
        
        return V
        
    def __getitem__(self, idx):
        """Gets items by loading from disk.
        
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        
        return self._load_item(self.file_list[idx])
    
    def _load_item(self, filepath):
        """ Load item based on filepath"""
        filepath = os.path.join(self.data_dir, filepath)
        V = self._load_file(filepath)
        
        seq = V["traj"][:,:4]  # lat, lon, sog, cog
        seq[seq>0.9999] = 0.9999 # cap extreme values
        seq = seq
        seq = torch.tensor(seq, dtype=torch.float32)
        
        seqlen = torch.tensor(seq.shape[0], dtype=torch.int)
        mmsi = torch.tensor(int(V["mmsi"]), dtype=torch.int)
        time_start = torch.tensor(V["traj"][0, 7], dtype=torch.int)
        
        return seq, seqlen, mmsi, time_start
    
    def get_sample_by_mmsi_and_start_time(self, mmsi: int, start_time: int):
        filepaths_for_mmsi = [f for f in self.file_list if int(f.split('_')[0])==mmsi]
        for filepath in filepaths_for_mmsi:
            item = self._load_item(filepath)
            if item[3] == start_time:
                return item
        raise KeyError(f"No samples found for {mmsi} with start time {start_time}")
    
    def get_sample_features(self, idx):
        """ Returns metadata features for a given sample index.
        
        Features:
           - Speed: speed_avg, speed_max, speed_std
           - Navigational status: nav_status
           - Temporal: duration, hour_start, month, season
           - Spatial: lat_max, lat_min, lon_max, lon_min, displacement, length_over_displacement, cog_std
        """
        filepath = os.path.join(self.data_dir, self.file_list[idx])
        V = self._load_file(filepath)
        traj = de_normalize_track(V["traj"])
        
        features = self._calculate_sample_metadata(traj)
        
        return features
        
    def _calculate_sample_metadata(self, traj):
        """ Calculate metadata features for a given trajectory. """
        
        features = {}
        
        # Speed features
        sog = traj[:,2]
        features['speed_avg'] = np.mean(sog)
        features['speed_max'] = np.max(sog)
        features['speed_std'] = np.std(sog)
        
        # Navigational status
        nav_status = traj[:,6]
        nav_status = nav_status[nav_status!=15] # 15 is unknown value
        if len(nav_status) == 0:
            features['nav_status'] = 15
        else:
            # Use mode
            features['nav_status'] = int(np.bincount(nav_status.astype(int)).argmax())
        
        # Temporal features
        time_start = traj[0,7]
        time_end = traj[-1,7]
        features['duration'] = time_end - time_start
        
        dt_start = datetime.fromtimestamp(time_start)
        features['hour_start'] = dt_start.hour
        features['month'] = dt_start.month
        
        # Spatial features
        latitudes = traj[:,0]
        longitudes = traj[:,1]
        features['lat_max'] = np.max(latitudes)
        features['lat_min'] = np.min(latitudes)
        features['lon_max'] = np.max(longitudes)
        features['lon_min'] = np.min(longitudes)
        
        displacement = geodesic((latitudes[0], longitudes[0]),(latitudes[-1], longitudes[-1])).meters
        features['displacement'] = displacement

        path = list(zip(latitudes, longitudes))
        length = sum(geodesic(p1, p2).meters for p1, p2 in zip(path, path[1:]))
        
        if displacement > 0:
            features['length_over_displacement'] = length / displacement
        else:
            features['length_over_displacement'] = 1.0  # Impute value
        
        features['cog_std'] = np.std(traj[:,3])
        
        return features
        