from MSD.settings.config import LABEL_FILE_TRAIN, LARGE_DATASET, DATASET_DESCRIPTION
import pandas as pd
import numpy as np
from scipy.io import loadmat
from PIL import Image
import os

def create_description_matrix():
    label_matrix = loadmat(LABEL_FILE_TRAIN)['annotations'].reshape(-1,1)

    file_names = [label_matrix[x][0][0][0].split('/')[1] for x in range(len(label_matrix))]
    img_classes = [label_matrix[x][0][5][0][0] for x in range(len(label_matrix))]
    is_test = [label_matrix[x][0][6][0][0] for x in range(len(label_matrix))]

    shapes = []
    for i in range(len(label_matrix)):
        img = Image.open(LARGE_DATASET + '/' + file_names[i])
        arr = np.array(img)
        shapes.append(arr.shape)

    shapes_df = pd.DataFrame(shapes, columns=['height', 'width', 'channels'])

    names_df = pd.DataFrame({'file_names':  file_names,
                             'class_id':    img_classes,
                             'is_test':     is_test})

    names_df['full_path'] = LARGE_DATASET + '/' + names_df.file_names

    names_df = pd.concat([names_df, shapes_df], axis='columns')

    names_df.to_csv(DATASET_DESCRIPTION, index=False)

create_description_matrix()

