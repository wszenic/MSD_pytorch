from MSD.settings.config import LABEL_FILE_TRAIN, TRAIN_FOLDER, DATASET_DESCRIPTION

import pandas as pd
from torchvision import transforms
from scipy.io import loadmat
from PIL import Image
import torch


class CarsDataset:

    def __init__(self, path_list, use_transforms=True):
        self.use_transforms = use_transforms
        self.image_paths = path_list

        labels_df = pd.read_csv(DATASET_DESCRIPTION)
        labels_df = labels_df[['preprocess_path', 'class_id']]
        labels_df = labels_df.set_index('preprocess_path')
        label_dict = labels_df.to_dict('index')

        self.labels = [label_dict[x]['class_id'] for x in self.image_paths]

        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_tensor = torch.load(self.image_paths[item])
        if self.use_transforms:
            image_tensor = self.transforms(image_tensor)
        label = torch.tensor(self.labels[item], dtype=torch.long)

        return image_tensor, label
