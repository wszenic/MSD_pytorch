from MSD.data.dataparser import CarsDataset
from MSD.settings.config import TRAIN_FOLDER
import glob
import numpy as np
import os
from PIL import Image


class DatasetDescriptor:
    def __init__(self, train_paths):
        self.train_paths = train_paths

    def calculate_mean_std(self):
        train_dataset = CarsDataset(self.train_paths)

        img_arrays = []
        for im in range(len(train_dataset)):
            if len(train_dataset[im][0]) == 3:
                img_array = np.array(train_dataset[im][0])
                img_arrays.append(img_array)

        img_arrays = np.array(img_arrays)

        mean_r = np.mean(img_arrays[0, :, :])
        mean_g = np.mean(img_arrays[1, :, :])
        mean_b = np.mean(img_arrays[2, :, :])

        std_r = np.std(img_arrays[0, :, :])
        std_g = np.std(img_arrays[1, :, :])
        std_b = np.std(img_arrays[2, :, :])

        return {'mean': (mean_r, mean_g, mean_b),
                'std': (std_r, std_g, std_b)}

    def remove_single_color(self):

        for im_path in self.train_paths:
            arr = np.array(Image.open(im_path))
            if len(arr.shape) < 3:
                print(f'shape found :{arr.shape}')
                os.remove(im_path)

# train_paths = glob.glob(TRAIN_FOLDER + '\\*.jpg')
# DatasetDescriptor(train_paths).remove_single_color()

# {'mean': (0.558926, 0.42738813, 0.4077543), 'std': (0.24672212, 0.236501, 0.22921552)}
