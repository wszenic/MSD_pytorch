from MSD.settings.config import LABEL_FILE_TRAIN, TRAIN_FOLDER

from torchvision import transforms
from scipy.io import loadmat
from PIL import Image
import torch


class CarsDataset:

    def __init__(self, path_list, use_transforms=True):
        """"
        label matrix structure:
        0-3 :   bounding boxes
        4   :   class_name
        5   :   file_name
        """
        self.use_transforms = use_transforms
        self.image_paths = path_list

        # dict -> filename : class name
        labels_list = loadmat(LABEL_FILE_TRAIN)['annotations'].reshape(-1, 1)
        label_dict = {labels_list[x][0][5][0]: labels_list[x][0][4][0][0] for x in range(len(labels_list))}

        self.labels = [label_dict[x.split('\\')[-1]] for x in path_list]

        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.558926, 0.42738813, 0.4077543),
                                 std=(0.24672212, 0.236501, 0.22921552))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        label = torch.tensor(self.labels[item], dtype=torch.long)

        image = Image.open(image_path)
        if self.use_transforms:
            image = self.transforms(image)

        return image, label
