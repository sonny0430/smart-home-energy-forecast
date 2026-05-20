import numpy as np
import torch
from torch.utils.data import Dataset


class EnergySequenceDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int = 24, target_col_idx: int = 0):
        self.data = data
        self.seq_len = seq_len
        self.target_col_idx = target_col_idx

        if len(self.data) <= self.seq_len:
            raise ValueError("데이터 길이가 seq_len보다 짧습니다.")

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len, self.target_col_idx]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )