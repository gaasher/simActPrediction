import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class SSLDataset(Dataset):
    def __init__(self, path, stage, seq_len=128, max=np.array([3.14159274e+00, 1.56960034e+00, 3.14159250e+00, 9.43980408e+00,
        3.51681976e+01, 1.13550110e+01, 4.67008477e+04, 8.15972363e+03,
        1.16144785e+04, 1.12840000e+04, 4.40000000e+03, 2.69869995e+02]), 
        min=np.array([-3.14159250e+00, -1.57079637e+00, -3.14159274e+00, -1.06317291e+01,
        -1.36996613e+01, -1.10269775e+01, -2.36590332e+03, -9.05344043e+03,
        -3.21341064e+03, -5.03000000e+02,  3.21600008e+00,  0.00000000e+00])):


        super().__init__()
        self.path = path
        self.max = max
        self.min = min
        self.seq_len = seq_len

        # open data from parquet
        self.data = pd.read_parquet(f'{self.path}/nih_ssl_{stage}.parquet')

        self.data = self.data.astype(np.float64) # shape is (num_samples, seq_len, num_channels)

        self.data_shape = self.data.shape
        print(f'Loaded {stage} data with shape {self.data.shape}')


    
        # find start indices, which are determined by looking at windows of length 128 with sliding window of legth 64 in unique file_ids (column 1), store 
        self.start_indices = []
        #create temporary column to store the index of each row
        for file_id in self.data['file_id'].unique():
            file_data = self.data[self.data['file_id'] == file_id]
            start_index = file_data.index[0]

            for i in range(0, len(file_data), 64):
                if i+seq_len <= len(file_data):
                    try:
                        self.start_indices.append(i+start_index)
                    except:
                        print(i+start_index)
    
        # turn data into a numpy array
        self.data = self.data.to_numpy()

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        index = self.start_indices[idx]
        # return last 12 column of data for index: index to index+128
        data =  self.data[index:index+self.seq_len, 2:]
        # normalize data
        data = (data - self.min) / (self.max - self.min)
        #clip data
        data = np.clip(data, 0, 1)
        return data