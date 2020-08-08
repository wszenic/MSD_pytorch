from MSD.settings.config import BATCH_SIZE, TRAIN_PERC, TRAIN_FOLDER
from MSD.data.dataparser import CarsDataset
from MSD.models.MSD_net import MSDnet
import torch.distributed as dist
from sklearn.model_selection import train_test_split
import glob
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
import torch
from torch.utils.data import DataLoader
from torch import nn


class NeuralNetworkLearner(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.batch_size = BATCH_SIZE

        self.model = MSDnet()
        self.loss = nn.CrossEntropyLoss()

    @pl.core.decorators.auto_move_data
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        x, label = batch
        x = self.forward(x)
        loss = self.loss(x, label)

        logs = {'train loss': loss}

        return {'loss': loss, 'log': logs}

    # def validation_step(self, batch, batch_id):
    #     x, label = batch
    #     x = self.forward(x)
    #     loss = self.loss(x, label)
    #
    #     return {'val loss': loss}

    def prepare_data(self):
        train_paths, valid_paths = train_test_split(glob.glob(TRAIN_FOLDER + '\\*.jpg')[:100], test_size=TRAIN_PERC)

        self.train_dataset = CarsDataset(train_paths)
        self.val_dataset = CarsDataset(valid_paths)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


