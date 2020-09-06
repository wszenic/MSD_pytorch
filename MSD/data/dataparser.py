from MSD.settings.config import LABEL_FILE_TRAIN, TRAIN_FOLDER, DATASET_DESCRIPTION,\
                                IMAGE_COLOUR_MODE

import pandas as pd
from scipy.io import loadmat
from PIL import Image
import torch


class CarsDataset:

    def __init__(self, path_list, transforms):
        self.transforms = transforms
        self.image_paths = path_list

        labels_df = pd.read_csv(DATASET_DESCRIPTION)
        labels_df = labels_df[['full_path', 'class_id']]
        labels_df = labels_df.set_index('full_path')
        label_dict = labels_df.to_dict('index')

        self.labels = [label_dict[x]['class_id'] - 1 for x in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img = Image.open(self.image_paths[item]).convert(IMAGE_COLOUR_MODE)
        image_tensor = self.transforms(img)
        label = torch.tensor(self.labels[item], dtype=torch.long)

        return image_tensor, label
