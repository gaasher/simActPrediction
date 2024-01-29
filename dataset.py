import numpy as np
import zarr
from torch.utils.data import Dataset

class simAPDataset(Dataset):
    def __init__(self, path, stage, transform=None):
        super().__init__()
        self.path = path
        self.stage = stage
        self.transform = transform # for now we will just use the raw data, but if we want to add augmentations

        self.data = zarr.open(f'{self.path}/{stage}_data.zarr', mode='r')
        self.labels = zarr.open(f'{self.path}/{stage}_labels.zarr', mode='r')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]


        # add normalization here
        if self.transform:
            x = self.transform(x)
        return x, y