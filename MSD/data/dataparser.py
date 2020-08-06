from MSD.settings.config import DATA_PATH, TRAIN_FOLDER, TEST_FOLDER, LABEL_FILE

from scipy.io import loadmat
from PIL import Image
from glob import glob
import numpy as np


class CarsDataset:

    def __init__(self, is_train=True):
        """"
        label file structure:
        0-3 :   bounding boxes
        4   :   class_name
        5   :   file_name
        """
        self.path = DATA_PATH + (TRAIN_FOLDER if is_train else TEST_FOLDER)
        self.image_paths = [x for x in glob(self.path + "\\*")]

        labels_list = loadmat(LABEL_FILE)['annotations'].reshape(-1,1)
        self.labels = np.array([labels_list[x][0][4] for x in range(len(labels_list))]).flatten()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        label = self.labels[item]

        image = Image.open(image_path)

        return image, label



