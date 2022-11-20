import librosa
import os
from torch.utils.data.dataset import Dataset
import torch

class NottinghamDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.name_list = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        arr, _ = librosa.load(os.path.join(self.data_dir, self.name_list[idx]))
        return torch.from_numpy(arr)