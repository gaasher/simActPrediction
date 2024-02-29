import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class NIHDataset(Dataset):
    def __init__(self, path, stage, do_transform=False, transforms=None, max=None, min=None):
        super().__init__()
        self.path = path
        self.stage = stage
        self.do_transform = do_transform
        self.transforms = ['noise', 'channel_off']

        # open data from numpy arrays
        self.data = np.load(f'{self.path}/nih_{self.stage}.npy', allow_pickle=True)
        self.data = self.data.astype(np.float64) # shape is (num_samples, seq_len, num_channels)
        if max is not None and min is not None:
            # open train data for normalization
            self.train_data = np.load(f'{self.path}/nih_train.npy', allow_pickle=True)
            self.train_data = self.train_data.astype(np.float64) # shape is (num_samples, seq_len, num_channels)

            print(f'Loaded {self.stage} data with shape {self.data.shape} and train data with shape {self.train_data.shape}')
            #normalize the data on 12 channels (get max and min of each of the 12 channels and normalize the data based on that)
            # Calculate max and min values for the first 12 channels
            max_val = np.max(self.train_data[:, :, :12], axis=(0, 1))
            min_val = np.min(self.train_data[:, :, :12], axis=(0, 1))
        else:
            print(f'max and min are: {max} and {min}')
            max_val = max
            min_val = min

        # Normalize only the first 12 channels
        self.data[:, :, :12] = (self.data[:, :, :12] - min_val) / (max_val - min_val)


    # create augmentation functions for 1) turn random channels into noise 2) add noise to random parts of the signal
    def channel_off(self, x, max_perc_off=0.3):
        #shape of x is (num_channels, 128)
        perc_off = np.random.uniform(0, max_perc_off)
        num_off = int(perc_off * x.shape[0])
        off_idx = np.random.choice(x.shape[0], num_off, replace=False)
        x[off_idx] = np.random.normal(0, 1, (num_off, 128))
        return x
    
    def add_noise(self, x, max_perc_noise=0.2):
        #shape of x is (num_channels, 128)
        perc_noise = np.random.uniform(0, max_perc_noise)
        noise_idx = np.random.choice(x.shape[1], int(perc_noise * x.shape[1]), replace=False)
        x[:, noise_idx] = np.random.normal(0, 1, (x.shape[0], len(noise_idx)))
        return x
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        # (seq_len, num_channels) -> (num_channels, seq_len)
        x = x[:, :12]
        x = np.transpose(x, (1, 0))

        #clip data
        x = np.clip(x, 0, 1)

        y = self.data[idx][0][-1]

        # add augmentation here
        if self.do_transform:
            if 'noise' in self.transforms:
                x = self.add_noise(x)
            if 'channel_off' in self.transforms:
                x = self.channel_off(x)
        # add normalization here
        
        return x, y