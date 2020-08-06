from MSD.settings.config import BATCH_SIZE, TRAIN_PERC, TRAIN_FOLDER, TEST_FOLDER
from MSD.data.dataparser import CarsDataset

from sklearn.model_selection import train_test_split
import glob
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch import nn


class NeuralNetworkLearner(pl.LightningModule):

    def __init__(self, model, transforms):
        super().__init__()

        self.save_hyperparameters()
        self.batch_size = BATCH_SIZE

        self.transforms = transforms
        self.model = model
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

    def validation_step(self, batch, batch_id):
        x, label = batch
        x = self.forward(x)
        loss = self.loss(x, label)

        return {'val loss': loss}

    def prepare_data(self):
        train_paths, valid_paths = train_test_split(glob.glob(TRAIN_FOLDER + '\\*.jpg'), test_size=TRAIN_PERC)
        test_paths = glob.glob(TEST_FOLDER + '\\*.jpg')

        train_dataset = CarsDataset(train_paths, 'train')
        val_dataset = CarsDataset(valid_paths, 'train')
        test_dataset = CarsDataset(test_paths, 'test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
