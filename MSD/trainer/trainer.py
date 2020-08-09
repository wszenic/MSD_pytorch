from MSD.settings.config import BATCH_SIZE, TRAIN_PERC, TRAIN_FOLDER, NEPTUNE_TOKEN
from MSD.data.dataparser import CarsDataset
from MSD.models.MSD_net import MSDnet

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch import nn

from sklearn.model_selection import train_test_split
import glob
import torch

import neptune

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
        neptune.log_metric('train loss', loss)
        return {'loss': loss, 'log': logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['train loss'] for x in outputs]).meanr()
        neptune.log_metric('train loss', avg_loss)

        return {'avg_train_loss': avg_loss}

    def validation_step(self, batch, batch_id):
        x, label = batch
        x = self.forward(x)
        loss = self.loss(x, label)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        neptune.log_metric('train loss', avg_loss)
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


    def prepare_data(self):
        file_paths = glob.glob(TRAIN_FOLDER + '\\*.jpg')

        train_paths, valid_paths = train_test_split(file_paths, test_size=TRAIN_PERC)

        self.train_dataset = CarsDataset(train_paths)
        self.val_dataset = CarsDataset(valid_paths)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


