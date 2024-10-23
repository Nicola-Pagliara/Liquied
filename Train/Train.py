import os
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from Dataloader.dataManager import TSDataset
"""
Find ModelCheckpoint module
"""


class Train:
    def __init__(self, model):
        self.model = model
        self.train = Trainer(accelerator='cpu', max_epochs=1, logger=WandbLogger(name='Liquied', log_model='all'),
                             callbacks=[], max_steps=100)
        self.dataset = TSDataset()

    def train(self, dataset_loader) -> None:
        self.train.fit(model=self.model, train_dataloaders=dataset_loader)
        return

    def validation(self, dataset_val_loader) -> None:
        self.train.validate(model=self.model, dataloaders=dataset_val_loader)
        return
