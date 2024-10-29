import numpy
import pytorch_lightning as L
import pandas as pd
import torch
import json
from torch.utils.data import random_split, DataLoader
from utils.Logger import getLogger

from Dataloader.ElConsDataset import ElConsDataset


class TSDataset(L.LightningDataModule):

    def __init__(self, batch_size, path_dir, timestep):
        super().__init__()
        self.test = None
        self.ecdl_train = None
        self.ecdl_val = None
        self.predict = None
        self.batch_size = batch_size
        self.path_dir = path_dir
        self.timestep = timestep

    def prepare_data(self) -> None:
        pass
        return

    def setup(self, stage: str) -> None:

        if stage == 'fit':
            with open(self.path_dir, 'r') as f:
                dataset = json.load(f)
                f.close()
            dataset = dataset['load of AT']
            dataset = numpy.asarray(dataset)
            ecdl_full = ElConsDataset(dataset=dataset, timestep=self.timestep)
            self.ecdl_train, self.ecdl_val = random_split(dataset=ecdl_full, lengths=[0.7, 0.3],
                                                          generator=torch.Generator().manual_seed(42))
        if stage == 'test':
            with open(self.path_dir, 'r') as f:
                dataset = json.load(f)
                f.close()
            dataset = dataset['load of LU']
            dataset = numpy.asarray(dataset)
            self.test = ElConsDataset(dataset=dataset, timestep=self.timestep)

        if stage == 'predict':
            with open(self.path_dir, 'r') as f:
                dataset = json.load(f)
                f.close()
            dataset = dataset['load of AT']
            dataset = numpy.asarray(dataset)
            ecdl_pred = ElConsDataset(dataset=dataset, timestep=self.timestep)
            self.predict = ecdl_pred

        return

    def train_dataloader(self):
        return DataLoader(self.ecdl_train, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.ecdl_val, batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=1)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=1)
