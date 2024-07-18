import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
import torchvision
import torchvision.transforms as transforms
from scipy.signal import butter, filtfilt, resample


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        print(self.X.shape)
        meanX = self.X.mean(dim=(1,2), keepdim=True)
        stdX = self.X.std(dim=(1,2), keepdim=True)
        for i in range(len(self.X)):
            self.X[i] = (self.X[i] - meanX[i]) / (stdX[i] + 10**(-6))
        # self.X = bandpass_filter(self.X.detach().numpy().copy(), 1.0, 40.0, 200)
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        print(self.subject_idxs.shape)

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
