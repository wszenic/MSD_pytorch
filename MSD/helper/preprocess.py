from MSD.settings.config import LARGE_DATASET, PREPROCESS_FOLDER, DATASET_DESCRIPTION
from PIL import Image
from tqdm import tqdm
import glob
from torchvision import transforms
import torch
import pandas as pd


class Cropper:

    def __init__(self):
        self.path = LARGE_DATASET

        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.558926, 0.42738813, 0.4077543),
                                 std=(0.24672212, 0.236501, 0.22921552))
        ])

    def crop_photos(self):
        labels_df = pd.read_csv(DATASET_DESCRIPTION)
        labels_df = labels_df[['file_names', 'channels']]
        labels_df = labels_df.set_index('file_names')

        labels_dict = labels_df.to_dict('index')

        for file in tqdm(glob.glob(LARGE_DATASET + '/*.jpg')):
            file_name = file.split('/')[-1]

            if labels_dict[file_name]['channels'] == 3.0:
                img = Image.open(file)
                out_tensor = self.transforms(img)
                torch.save(out_tensor, PREPROCESS_FOLDER + '/' + file_name.split('.')[0] + '.pt')



cropper = Cropper()
cropper.crop_photos()
