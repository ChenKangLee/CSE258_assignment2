import numpy as np
import torch
from torch.utils.data import Dataset

class BPRDataset(Dataset):
    def __init__(self, path):
        self.data = np.loadtxt(path)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]