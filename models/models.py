import torch.nn as nn
import torch.optim as optim
from common.config import DEVICE
from common.datasets import DatasetConUnParametro
from .base import ModeloBase  # if exists

class ModeloDropout(ModeloBase):
    def __init__(self, capas, batch_size, lr, dropout, indice, carpeta, desc):
        super().__init__(batch_size, lr, indice, carpeta, desc)
        layers = []
        prev = 66
        for c in capas:
            layers += [nn.Linear(prev, c), nn.ReLU(), nn.Dropout(dropout)]
            prev = c
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, x): return self.model(x)

# Instantiate models here or in run_all_models