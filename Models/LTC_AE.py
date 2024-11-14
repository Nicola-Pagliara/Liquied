from collections import OrderedDict

import numpy
from ncps.torch import LTC, LTCCell
from ncps.wirings import AutoNCP
from torch import nn
import pytorch_lightning as L
import torch
import torch.nn.functional as F
import torchmetrics as metrics
import math


class LTCAutoEncoder(L.LightningModule):
    def __init__(self, input_dim, encoding_dim, lr):
        super().__init__()
        self.lr = lr
        self.encode = LTC(input_size=input_dim, units=AutoNCP(units=encoding_dim * 2, output_size=encoding_dim),
                          return_sequences=False, batch_first=True)

        self.decode = LTCCell(in_features=encoding_dim,
                              wiring=AutoNCP(units=encoding_dim * 2, output_size=encoding_dim))

        self.outproj = torch.nn.Linear(encoding_dim, input_dim)
        self.loss_fn = torch.nn.MSELoss()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x = batch
        # x = x.view(x.size(0), -1)
        decoded, encoded = self(x)
        loss = self.loss_fn(decoded, x)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        # x = x.view(x.size(0), -1)
        decoded, encoded = self(x)
        loss = self.loss_fn(decoded, x)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x_target = batch
        decoded_test, _ = self(x_target)
        mse_test = metrics.regression.MeanSquaredError().to(decoded_test)
        mae_test = metrics.regression.MeanAbsoluteError().to(decoded_test)
        reconstruction_error = mse_test(decoded_test, x_target)
        mae_red_error = mae_test(decoded_test, x_target)
        rmse_test = math.sqrt(reconstruction_error)
        std = numpy.asarray(x_target).std()
        norm_rmse = rmse_test / std
        self.log('MSE test', reconstruction_error, prog_bar=True)
        self.log('RMSE test', rmse_test, prog_bar=True)
        self.log('MAE test', mae_red_error, prog_bar=True)
        self.log('NRMSE std test', norm_rmse, prog_bar=True)
        return reconstruction_error, mae_test, rmse_test, norm_rmse

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        decoded, encoded = self(x)
        return {'decoded': decoded, 'encoded': encoded}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.2, patience=30,
                                                               min_lr=1e-7)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def forward(self, x):
        # assert x.dtype == torch.float32
        encoded, _ = self.encode(x)

        outputs = []
        encoded_next = encoded
        states = torch.zeros(x.size(0), self.decode.state_size).to(encoded_next)
        for t in range(0, x.shape[1]):
            encoded_next, states = self.decode(encoded_next, states)
            outputs.append(self.outproj(encoded_next))
        decoded = torch.stack(outputs, dim=1)
        return decoded, encoded


class LTC_AE_normal(L.LightningModule):
    def __init__(self, input_dim, encoding_dim, lr):
        super().__init__()
        self.lr = lr
        self.encode = nn.Sequential(
            LTC(input_size=input_dim, units=encoding_dim, return_sequences=False)

        )

        self.decode = nn.Sequential(
            LTC(input_size=input_dim, units=encoding_dim, return_sequences=False),
            # nn.Linear(encoding_dim, input_dim)
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x = batch
        x = x.view(x.size(0), -1)
        decoder, encoder = self.forward(x)
        loss = F.mse_loss(decoder, x)
        self.log("train_loss", loss, prog_bar=True)
        return

    def validation_step(self, batch, batch_idx):
        x = batch
        x = x.view(x.size(0), -1)
        decoder, encoder = self.forward(x)
        loss = F.mse_loss(decoder, x)
        self.log("val_loss", loss, prog_bar=True)
        return

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.2, patience=30,
                                                               min_lr=1e-7)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}

    def forward(self, x):
        # assert x.type == torch.float64
        encoded = self.encode(x)
        # print(encoded[0], encoded[0].shape)
        decoded = self.decode(encoded[0])
        return decoded, encoded
