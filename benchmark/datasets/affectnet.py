import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class Affectnet(Dataset):
    def __init__(self, df, split='train', transform=None, target_transform=None, ):
        self.df = df[df.split == split]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx].image_path

        image = read_image(img_path)
        label = self.df.iloc[idx].label
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        image = image.permute([1, 2, 0]).numpy()
        return image, label


class FaceDataset(Dataset):

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