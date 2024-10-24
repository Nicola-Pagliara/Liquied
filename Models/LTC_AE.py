from collections import OrderedDict
from ncps.torch import LTC
from torch import nn
import pytorch_lightning as L
import torch
import torch.nn.functional as F


class LTC_AutoEncoder(L.LightningModule):
    def __init__(self, input_dim, encoding_dim, lr):
        super().__init__()
        self.lr = lr
        self.encode = nn.Sequential(
            LTC(input_dim, encoding_dim)
        )

        self.decode = nn.Sequential(
            LTC(encoding_dim, input_dim),
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x = batch
        x = x.view(x.size(0), 1)
        decoder, encoder = self.forward(x)
        loss = F.mse_loss(decoder[0], x)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return

    def validation_step(self, batch, batch_idx):
        x = batch
        x = x.view(x.size(0), 1)
        decoder, encoder = self.forward(x)
        loss = F.mse_loss(decoder[0], x)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.2, patience=30,
                                                               min_lr=1e-7)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}

    def forward(self, x):
        # assert x.dtype == torch.float64
        _, encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded
