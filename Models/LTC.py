import os
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from collections import OrderedDict
from ncps.torch import LTC


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

    def training_step(self, batch, batch_idx):

        return

    def validation_step(self, *args: Any, **kwargs: Any):
        return

    def test_step(self, *args: Any, **kwargs: Any):
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        batch, timesteps, features = x.size()
        x = self.input(x)
        x = self.in_activation(x)
        x = self.hiddens(x)
        out, liq_state = self.liquid(x)
        out = self.output(out)
        out = torch.log_softmax(out, dim=1)
        return out
