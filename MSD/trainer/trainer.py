from MSD.settings.config import BATCH_SIZE, TRAIN_PERC, TRAIN_FOLDER,\
                                NEPTUNE_TOKEN, PREPROCESS_FOLDER,\
                                DATASET_DESCRIPTION
from MSD.data.dataparser import CarsDataset
from MSD.models.MSD_net import MSDnet

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import pandas as pd

import torchvision
from torchvision import transforms

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

        neptune.create_experiment('MSD')

    @pl.core.decorators.auto_move_data
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        x, label = batch

        class1_y, class2_y, class3_y = self.forward(x)

        loss_class_1 = self.loss(class1_y, label).view(1)
        loss_class_2 = self.loss(class2_y, label).view(1)
        loss_class_3 = self.loss(class3_y, label).view(1)

        loss = torch.mean(torch.cat([loss_class_1, loss_class_2, loss_class_3]))

        logs = {'train_loss': loss}

        return {'loss': loss, 'log': logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        neptune.log_metric('avg_train_loss', avg_loss)

        return {'avg_train_loss': avg_loss}

    def validation_step(self, batch, batch_id):
        x, label = batch

        class1_y, class2_y, class3_y = self.forward(x)

        loss_class_1 = self.loss(class1_y, label).view(1)
        loss_class_2 = self.loss(class2_y, label).view(1)
        loss_class_3 = self.loss(class3_y, label).view(1)

        loss = torch.mean(torch.cat([loss_class_1, loss_class_2, loss_class_3]))

        logs = {'val_loss': loss}

        return {'loss': loss, 'log': logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        neptune.log_metric('val_loss', avg_loss)

        return {'val_loss': avg_loss}


    def prepare_data(self):
        dataset_description = pd.read_csv(DATASET_DESCRIPTION)

        dataset_description = dataset_description[dataset_description.channels == 3]
        dataset_description = dataset_description[dataset_description.is_test == 0]

        train_x, val_x, _, _ = train_test_split(dataset_description.full_path.tolist(),
                                                dataset_description.class_id.tolist(), train_size=0.5,
                                                stratify=dataset_description.class_id.tolist())

        self.train_dataset = CarsDataset(train_x)
        self.val_dataset = CarsDataset(val_x)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=6, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=6, pin_memory=True)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        return [optimizer], [scheduler]


