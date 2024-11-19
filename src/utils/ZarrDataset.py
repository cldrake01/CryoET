import torch
from torch.utils.data import Dataset
import zarr

class ZarrDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.z = zarr.open(path)

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return torch.tensor(self.z[idx])
