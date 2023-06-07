import numpy as np
import pandas as pd
import torch

# additional helper functions
ROWS_PER_FRAME = 543
max_length = 80

def load_relevant_data_subset(pq_path, type='parquet'):
    data_columns = ['x', 'y', 'z']
    if type=='parquet':    
        data = pd.read_parquet(pq_path, columns=data_columns)
    else:
        data = pd.read_csv(pq_path, usecols=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def pre_process(xyz):
    #xyz = xyz - xyz[~torch.isnan(xyz)].mean(0,keepdims=True) #noramlisation to common maen
    #xyz = xyz / xyz[~torch.isnan(xyz)].std(0, keepdims=True)
    
    LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ]
    
    lip   = xyz[:, LIP]
    lhand = xyz[:, 468:489]
    rhand = xyz[:, 522:543]
    xyz = torch.cat([ #(none, 82, 3)
        lip,
        lhand,
        rhand,
    ],1)
    xyz[torch.isnan(xyz)] = 0
    xyz = xyz[:max_length]
    return xyz