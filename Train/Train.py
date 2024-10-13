import os
from lightning import Trainer


class Train:
    def __init__(self, model):
        self.model = model
        self.train = Trainer(accelerator='cpu', max_epochs=50)

    def train(self, dataset_loader) -> None:
        self.train.fit(model=self.model, train_dataloaders=dataset_loader)
        return

    def validation(self, dataset_val_loader) -> None:
        self.train.validate(model=self.model, dataloaders=dataset_val_loader)
        return

