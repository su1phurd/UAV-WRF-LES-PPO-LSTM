# data_loader.py
import numpy as np
from netCDF4 import Dataset

def load_raw_sequences(nc_path):
    """直接加载原始浓度序列和源头浓度"""
    with Dataset(nc_path, 'r') as nc:
        sequences = []
        source_concs = []
        
        for ep in range(len(nc['episode'])):
            steps = np.where(~np.isnan(nc['x'][ep]))[0]
            if len(steps) == 0:
                continue
                
            conc_seq = nc['concentration'][ep, :steps[-1]+1].tolist()
            source_conc = nc['source_concentration'][ep]
            
            sequences.append(conc_seq)
            source_concs.append(source_conc)
    
    return sequences, np.array(source_concs)