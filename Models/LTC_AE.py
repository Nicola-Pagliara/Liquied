from collections import OrderedDict
from ncps.torch import LTC
from torch import nn
import lightning as L
import torch
import torch.nn.functional as F


class LTC_AutoEncoder(L.LightningModule):
    def __int__(self, input_dim, encoding_dim, lr):
        self.lr = lr
        self.encode = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SELU(),
            LTC(64, encoding_dim)
        )

        self.decode = nn.Sequential(
            LTC(encoding_dim, 64),
            nn.SELU(),
            nn.Linear(64, input_dim)
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred, _ = self.forward(x)
        y_pred = y_pred.view_as(y)
        loss = self.binary_cross_entropy(y_pred, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return {"loss": loss},

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred, _ = self.forward(x)
        y_pred = y_pred.view_as(y)
        loss = self.binary_cross_entropy(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        batch, features = x.size()
        x = x.view(batch, -1)
        assert x.dtype == torch.float64
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded

