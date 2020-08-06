from MSD.settings.config import LABEL_FILE_TRAIN, LABEL_FILE_TEST

from scipy.io import loadmat
from PIL import Image
import numpy as np


class CarsDataset:

    def __init__(self, path_list, dataset_type):
        """"
        label matrix structure:
        0-3 :   bounding boxes
        4   :   class_name
        5   :   file_name
        """
        self.image_paths = path_list

        label_file = LABEL_FILE_TRAIN if dataset_type in ('train', 'valid') else LABEL_FILE_TEST
        labels_list = loadmat(label_file)['annotations'].reshape(-1, 1)

        label_dict = {labels_list[x][0][5][0]: labels_list[x][0][4][0][0] for x in range(len(labels_list))}

        self.labels = [label_dict[x.split('\\')[-1]] for x in path_list]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        label = self.labels[item]

        image = Image.open(image_path)
        #im_tensor = self.transforms(image)

        return image, image_path, label


