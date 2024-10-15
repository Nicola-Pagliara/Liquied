from collections import OrderedDict
from ncps.torch import LTC
from torch import nn
import lightning as L
import torch
import torch.nn.functional as F


class LTCNet(L.LightningModule):
    def __init__(self, in_features, out_features, n_hidden_layers, n_hidden, lr):
        super().__init__()
        self.learning_rate = lr
        self.input = nn.Linear(in_features, n_hidden)
        self.in_activation = nn.ReLU()
        hidden_layers = OrderedDict()
        for i in range(0, n_hidden_layers):
            hidden_layers['Linear' + str(i)] = nn.Linear(n_hidden, n_hidden)
            hidden_layers['Activation' + str(i)] = nn.ReLU()
        self.hiddens = nn.Sequential(hidden_layers)
        self.liquid = LTC(n_hidden, n_hidden, return_sequences=True)
        self.output = nn.Linear(n_hidden, out_features)
        self.save_hyperparameters()

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

    def binary_cross_entropy(self, logits, labels):
        return F.binary_cross_entropy(logits, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        batch, features = x.size()
        x = x.view(batch, -1)
        x = self.input(x)
        x = self.in_activation(x)
        x = self.hiddens(x)
        out, liq_state = self.liquid(x)
        out = self.output(out)
        out = F.log_softmax(out, dim=1)
        return out
