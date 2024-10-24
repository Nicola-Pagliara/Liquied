import pytorch_lightning as L
import pandas as pd
import torch
import json
from torch.utils.data import random_split, DataLoader
from utils.Logger import getLogger

from Dataloader.ElConsDataset import ElConsDataset


class TSDataset(L.LightningDataModule):

    def __init__(self, batch_size, path_dir):
        super().__init__()
        self.ecdl_train = []
        self.ecdl_val = []
        self.batch_size = batch_size
        self.path_dir = path_dir

    def prepare_data(self) -> None:
        pass
        return

    def setup(self, stage: str) -> None:
        with open(self.path_dir, 'r') as f:
            dataset = json.load(f)
            f.close()
        dataset = dataset['load of AT']
        ecdl_full = ElConsDataset(dataset=dataset, batch=self.batch_size, timestep=1)
        if stage == 'fit':
            self.ecdl_train, self.ecdl_val = random_split(dataset=ecdl_full, lengths=[0.7, 0.3],
                                                          generator=torch.Generator().manual_seed(42))

        return

    def train_dataloader(self):
        return DataLoader(self.ecdl_train, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.ecdl_val, batch_size=self.batch_size, num_workers=1)
