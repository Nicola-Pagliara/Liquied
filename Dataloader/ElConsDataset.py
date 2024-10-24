import torch
from torch.utils.data import Dataset


class ElConsDataset(Dataset):

    def __init__(self, dataset, timestep, batch):
        self.data = dataset
        self.batch = batch
        self.timestep = timestep

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = torch.tensor(sample)# not c
        sample = sample.reshape([self.batch, self.timestep])

        return sample.type(dtype=torch.LongTensor)
