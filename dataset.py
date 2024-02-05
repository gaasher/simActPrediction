import numpy as np
import zarr
from torch.utils.data import Dataset

class simAPDataset(Dataset):
    def __init__(self, path, stage, do_transform=False, transforms=None):
        super().__init__()
        self.path = path
        self.stage = stage
        self.do_transform = do_transform
        self.transforms = ['noise', 'channel_off']

        # open data from numpy arrays
        self.data = np.load(f'{self.path}/{self.stage}_data.npy')
        self.labels = np.load(f'{self.path}/{self.stage}_labels.npy')

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
    
    # add normalizer, unit norm for now
    def normalize(self, x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        # (seq_len, num_channels) -> (num_channels, seq_len)
        x = np.transpose(x, (1, 0))

        y = self.labels[idx]

        x = self.normalize(x)

        # add augmentation here
        if self.do_transform:
            if 'noise' in self.transforms:
                x = self.add_noise(x)
            if 'channel_off' in self.transforms:
                x = self.channel_off(x)
        # add normalization here
        
        return x, y