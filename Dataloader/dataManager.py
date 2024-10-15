import lightning as L
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from ElConsDataset import ElConsDataset


class TSDataset(L.LightningDataModule):

    def __init__(self, batch_size, path_dir):
        super().__init__()
        self.batch_size = batch_size
        self.path_dir = path_dir
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self) -> None:
        dataset = pd.read_excel(self.path_dir)
        dataset_filter = dataset['']
        dataset_filter = dataset_filter.dropna()
        dataset_filter.to_excel(self.path_dir)
        return

    def setup(self, stage: str) -> None:
        dataset = pd.read_excel(self.path_dir)
        ecdl_full = ElConsDataset(dataset=dataset, transform=self.transform)
        if stage == 'fit':
            self.ecdl_train, self.ecdl_val = random_split(dataset=ecdl_full, lengths=[0.7, 0.3], generator=torch.Generator().manual_seed(42))

        return

    def train_dataloader(self) :
        return DataLoader(self.ecdl_train, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.ecdl_val, batch_size=self.batch_size, num_workers=1)


