import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset


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
        return image, label

