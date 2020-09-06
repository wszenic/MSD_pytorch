from MSD.settings.config import BATCH_SIZE, TRAIN_PERC, TRAIN_FOLDER, NEPTUNE_TOKEN, PREPROCESS_FOLDER, \
    DATASET_DESCRIPTION, MAX_EPOCH, SCALE_CHANNELS, TRAIN_PERC, LEARNING_RATE, CLASSIFIER_SCALES, \
    USE_SCHEDULER, OPTIMIZER_STEP_SIZE, OPTIMIZER_GAMMA, IMAGE_COLOUR_MODE

from MSD.data.dataparser import CarsDataset
from MSD.models.MSD_net import MSDnet

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
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
        self.learning_rate = LEARNING_RATE

        self.model = MSDnet()
        self.loss = nn.CrossEntropyLoss()

        if IMAGE_COLOUR_MODE == 'RGB':
            normalization = transforms.Normalize(mean=(0.558926, 0.42738813, 0.4077543),
                                                 std=(0.24672212, 0.236501, 0.22921552))
        else:
            normalization = transforms.Normalize(mean=0.4646894,
                                                 std=0.23)

        self.train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomAffine(25, translate=(0.1, 0.1), scale=(0.9, 0.1), shear=8),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
            normalization,
            transforms.RandomHorizontalFlip(0.5)
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalization
        ])


        params = {'max epoch': MAX_EPOCH,
                  'channel scales': SCALE_CHANNELS,
                  'learning rate': LEARNING_RATE,
                  'classifier scale': CLASSIFIER_SCALES['out_ch'],
                  'training percentage': TRAIN_PERC,
                  'scheduler': USE_SCHEDULER,
                  'optimizer step size': OPTIMIZER_STEP_SIZE,
                  'optimizer gamma': OPTIMIZER_GAMMA,
                  'image colour mode (RGB / L)': IMAGE_COLOUR_MODE
                  }

        neptune.create_experiment('MSD', params=params)

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

        accuracy_class_1 = accuracy(torch.argmax(class1_y, dim=1), label).view(1)
        accuracy_class_2 = accuracy(torch.argmax(class2_y, dim=1), label).view(1)
        accuracy_class_3 = accuracy(torch.argmax(class3_y, dim=1), label).view(1)

        acc = torch.mean(torch.cat([accuracy_class_1, accuracy_class_2, accuracy_class_3]))

        logs = {'train_loss': loss}

        return {'loss': loss, 'acc': acc, 'log': logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        neptune.log_metric('avg_train_loss', avg_loss)

        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        neptune.log_metric('avg_train_acc', avg_acc)

        return {'avg_train_loss': avg_loss}

    def validation_step(self, batch, batch_id):
        x, label = batch

        class1_y, class2_y, class3_y = self.forward(x)

        loss_class_1 = self.loss(class1_y, label).view(1)
        loss_class_2 = self.loss(class2_y, label).view(1)
        loss_class_3 = self.loss(class3_y, label).view(1)

        loss = torch.mean(torch.cat([loss_class_1, loss_class_2, loss_class_3]))

        accuracy_class_1 = accuracy(torch.argmax(class1_y, dim=1), label).view(1)
        accuracy_class_2 = accuracy(torch.argmax(class2_y, dim=1), label).view(1)
        accuracy_class_3 = accuracy(torch.argmax(class3_y, dim=1), label).view(1)

        acc = torch.mean(torch.cat([accuracy_class_1, accuracy_class_2, accuracy_class_3]))

        logs = {'val_loss': loss}

        return {'loss': loss, 'acc': acc, 'log': logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        neptune.log_metric('val_loss', avg_loss)

        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        neptune.log_metric('avg_val_acc', avg_acc)

        return {'val_loss': avg_loss}


    def prepare_data(self):
        dataset_description = pd.read_csv(DATASET_DESCRIPTION)

        dataset_description = dataset_description[dataset_description.channels == 3]
        dataset_description = dataset_description[dataset_description.is_test == 0]

        train_x, val_x, _, _ = train_test_split(dataset_description.full_path.tolist(),
                                                dataset_description.class_id.tolist(), train_size=TRAIN_PERC,
                                                stratify=dataset_description.class_id.tolist())

        self.train_dataset = CarsDataset(train_x, transforms=self.train_transforms)
        self.val_dataset = CarsDataset(val_x, transforms=self.val_transforms)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=6, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=6, pin_memory=True)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if USE_SCHEDULER:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=OPTIMIZER_STEP_SIZE, gamma=OPTIMIZER_GAMMA)
            return [optimizer], [scheduler]
        else:
            return optimizer


