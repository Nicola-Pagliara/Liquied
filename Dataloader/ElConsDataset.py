import numpy
import numpy as np
import torch
from torch.utils.data import Dataset


class ElConsDataset(Dataset):

    def __init__(self, dataset, timestep=200, window_size=1):
        self.data = dataset
        self.timestep = timestep
        self.window = window_size
        self.mean = numpy.mean(dataset)
        self.std = numpy.std(dataset)

    def __len__(self):
        return len(self.data) // self.timestep

    def __getitem__(self, idx):

        sample = self.data[self.timestep * idx: self.timestep * (idx + 1)]
        sample = (sample - self.mean) / (self.std + 1e-8)
        """
         sequences = []
        for i in range(len(sample) - self.window):
            sequences.append(self.data[i:i + self.window])

        sample = np.array(sequences)
        """
        sample = torch.tensor(sample, dtype=torch.float32)
        sample = sample.reshape([self.timestep, 1])
        return sample
