"""Customized Pytorch Dataset."""

import numpy as np
import os
import pickle
import joblib
import torch
from torch.utils.data import Dataset

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
            if f.endswith(file_extension)
        ]
        self.file_list.sort()

    def __len__(self):
        return len(self.file_list)
    
    def _load_file(self, filepath):
        """Load a single data file. Modify based on your file format."""
        # For pickle files:
        """with open(filepath, 'rb') as f:
            V = pickle.load(f)"""
        # For joblib files:
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
        # Load data from disk
        filepath = os.path.join(self.data_dir, self.file_list[idx])
        V = self._load_file(filepath)
        
        seq = V["traj"][:,:4]  # lat, lon, sog, cog
        seq[seq>0.9999] = 0.9999 # cap extreme values
        seq = seq
        seq = torch.tensor(seq, dtype=torch.float32)
        
        seqlen = torch.tensor(seq.shape[0], dtype=torch.int)
        mmsi = torch.tensor(int(V["mmsi"]), dtype=torch.int)
        time_start = torch.tensor(V["traj"][0, 7], dtype=torch.int)
        
        return seq, seqlen, mmsi, time_start