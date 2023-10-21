import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class FaceDataset(Dataset):
    # https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=icml_face_data.csv
    def __init__(self, df, transform=None, ):

        self.df = df.to_dict('records')

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df[idx]

        label = row['emotion']
        pixels = row[' pixels']

        pixels = np.array(pixels.split()).astype(int)
        pixels = np.reshape(pixels, (48, 48))
        pixels = np.expand_dims(pixels, axis=-1)
        image = np.repeat(pixels, 3, axis=-1)
        image = np.uint8(image)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label