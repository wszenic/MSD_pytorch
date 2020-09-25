from MSD.settings.config import DATASET_DESCRIPTION
from torchvision import transforms
from PIL import Image
import pandas as pd

class Evaluator:

    def __init__(self, model):
        self.model = model
        self.predict_transforms = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.558926, 0.42738813, 0.4077543),
                                             std=(0.24672212, 0.236501, 0.22921552))
        ])
    def make_prediction(self, image_path):
        model = self.model
        image = Image.open(image_path)
        image_tensor = self.predict_transforms(image)
        predictions = model(image_tensor)

        return predictions

    def score_on_desription_csv(self, csv):
        df = pd.read_csv(DATASET_DESCRIPTION)

        df['scores'] = df.full.path.applu(self.make_prediction)

        df.to_csv('./output.csv')

