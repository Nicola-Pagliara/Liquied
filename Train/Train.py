import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from Dataloader.dataManager import TSDataset


class Train:
    def __init__(self, model):
        self.model = model
        self.train = Trainer(accelerator='cpu', max_epochs=1, logger=WandbLogger(name='Liquied', log_model='all'),
                             callbacks=[ModelCheckpoint(dirpath='Models/autoencoder_weights')], enable_checkpointing=True)
        self.dataset = TSDataset(batch_size=16, path_dir='Train/train_data/15min/AT_anomaly.json')

    def train(self) -> None:
        dataset_loader = self.dataset.train_dataloader()
        self.train.fit(model=self.model, train_dataloaders=dataset_loader)
        return

    def validation(self) -> None:
        dataset_val_loader = self.dataset.val_dataloader()
        self.train.validate(model=self.model, dataloaders=dataset_val_loader)
        return
