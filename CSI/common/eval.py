from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from common.common import parse_args
import models.classifier as C
from datasets import get_dataset, get_superclass_list, get_subclass_dataset
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import sys 
import os
from pathlib import Path
import cv2


P = parse_args()

class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class FaceDataset(Dataset):
    
    def __init__(self, df, split = 'T', transform=None, classes = [0, 2, 4, 6, 3, 5, 1]): 
        
        assert split in ['Train', 'Test', 'T'] 

        self.split = split
        
        self.classes = classes 
        
        self.df = df.loc[df[' Usage'].str.contains(split)]
        self.df = self.df.loc[df['emotion'].isin(classes)]
        self.df = self.df.to_dict('records')

        #print(len(self.df))

        self.data = []
        self.targets= []

        for idx in range(len(self.df)):
            row = self.df[idx]
            label = row['emotion']
            pixels = row[' pixels']

            pixels = np.array(pixels.split()).astype(int) 
            pixels = np.reshape(pixels, (48, 48))
            pixels = np.expand_dims(pixels, axis = -1)    
            image = np.repeat(pixels, 3, axis = -1)
            image = np.uint8(image) 

            self.data.append(image)
            self.targets.append(label)

        #print("data and targets:")
        #print(len(self.data))
        #print(len(self.targets))

        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
        row = self.df[idx]
        

        label = row['emotion']
        pixels = row[' pixels']

        pixels = np.array(pixels.split()).astype(int) 
        pixels = np.reshape(pixels, (48, 48))
        pixels = np.expand_dims(pixels, axis = -1)    
        image = np.repeat(pixels, 3, axis = -1)
        image = np.uint8(image)  
        '''
        #print(index)
        image, label = self.data[index], self.targets[index]    
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
 
        return image, label

class Food101Dataset(Dataset):
    
    def __init__(self, df, class_map = {}, split = 'T', transform=None, classes = []): 
        
        assert split in ['Train', 'Test', 'T'] 

        self.split = split
        
        self.classes = classes 
        self.fmap = class_map

        self.df = df.loc[df['Usage'].str.contains(split)]
        
        self.df[['Label', 'File']] = self.df.path.str.split('/', expand=True)
        self.df['Label'] = self.df['Label'].map(self.fmap)
        self.df['FilePath'] = '../../datasets/Food101/food-101/images/' + self.df['path']+'.jpg'
        #print(self.df.head(5))
        self.df = self.df.loc[self.df['Label'].isin(classes)]
        self.df = self.df.to_dict('records')
        #print(len(self.df))

        self.data = []
        self.targets= []

        for idx in range(len(self.df)):
            row = self.df[idx]
            label = row['Label']
            pixels = cv2.imread(row['FilePath'])

            pixels = cv2.resize(pixels, (48, 48), interpolation=cv2.INTER_CUBIC)
            image = np.uint8(pixels) 

            self.data.append(image)
            self.targets.append(label)

        #print("data and targets:")
        #print(len(self.data))
        #print(len(self.targets))

        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]    
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
 
        return image, label


class StanfordCarsDataset(Dataset):
    
    def __init__(self, df, class_map = {}, split = 'T', transform=None, classes = []): 
        
        assert split in ['Train', 'Test', 'T'] 

        self.split = split
        
        self.classes = classes 
        self.fmap = class_map

        self.df = df.loc[df['Usage'].str.contains(split)]
        self.df = self.df.loc[self.df['class'].isin(classes)]
        self.df['filepath'] = '../../datasets/cars' + self.df['filepath']
        self.df = self.df.to_dict('records')

        self.data = []
        self.targets= []

        for idx in range(len(self.df)):
            row = self.df[idx]
            label = int(row['class'])
            #print(row['FilePath'])
            pixels = cv2.imread(row['filepath'])
            #print(pixels.shape)
            x1, x2, y1, y2 = row['bbox_x1'], row['bbox_x2'], row['bbox_y1'], row['bbox_y2']
            b_pixels = pixels[x1:x2, y1:y2]
            #print(np.sum(b_pixels))
            if np.sum(b_pixels) == 0:
                b_pixels = pixels
            #print(b_pixels.shape)
            b_pixels = cv2.resize(b_pixels, (48, 48), interpolation=cv2.INTER_CUBIC)
            image = np.uint8(b_pixels) 

            self.data.append(image)
            self.targets.append(label)

        #print("data and targets:")
        #print(len(self.data))
        #print(len(self.targets))

        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]     
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
 
        return image, label

def get_dataloader(df, batch_size=32, normalize=False, size=32, in_classes = [0, 1, 2, 3, 4]): 

    fmap = {}
    if P.dataset == 'food':

        all_classes = [i for i in range(101)]
        in_classes = [i for i in range(90)]
        out_classes = [x for x in all_classes if x not in in_classes]
        train_df = pd.read_csv('../../datasets/Food101/food-101/meta/train.txt', 
        names=['path'], header=None) 
        train_df['Usage'] = 'Train'
        test_df = pd.read_csv('../../datasets/Food101/food-101/meta/test.txt', 
        names=['path'], header = None)
        test_df['Usage'] = 'Test'
        df = pd.concat([train_df, test_df], ignore_index = True)
        classes = os.listdir('../../datasets/Food101/food-101/images')
        for i, c in enumerate(classes):
            fmap[c] = i
            fmap[i] = c 

    elif P.dataset == 'face':
    
        all_classes = [0, 2, 4, 6, 3, 5, 1]
        in_classes = [5] #[0,1,2,3,4]
        out_classes = [x for x in all_classes if x not in in_classes]
    elif P.dataset == 'car':
        #size = 32
        all_classes = [i for i in range(196)]
        in_classes = [i for i in range(186)]
        out_classes = [x for x in all_classes if x not in in_classes]
        class_file = pd.read_csv('../../datasets/cars/car_classes.csv')
        class_file = class_file.rename(columns=class_file.iloc[0]).loc[1:]
        train_config_file = pd.read_csv('../../datasets/cars/train_cars_config.csv')
        train_config_file['Usage'] = 'Train'
        print(train_config_file.head(5))
        test_config_file = pd.read_csv('../../datasets/cars/test_cars_config.csv')
        test_config_file['Usage'] = 'Test'
        print(test_config_file.head(5))
        df = pd.concat([train_config_file, test_config_file], ignore_index=True)
 
    '''
    transform_train = [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
    
    transform_test = [transforms.Resize(size), transforms.ToTensor()]  
    '''
    transform_train = [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    transform_test = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]
    
    norm_layer = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
        )
    
    if normalize:
        transform_train.append(norm_layer)
        transform_test.append(norm_layer)
    
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)
    
    if P.mode == "simclr" or P.mode == "simclr_CSI":
        transform_train = TwoCropTransform(transform_train)
    
    if P.dataset == 'face':
        train_set = FaceDataset(df, split = 'Train', transform = transform_train, classes = in_classes)     
        test_set = FaceDataset(df, split = 'Test', transform = transform_test, classes = in_classes)     
        ood_set = FaceDataset(df, split = 'T', transform = transform_test, classes = out_classes)   
    elif P.dataset == 'food':
        train_set = Food101Dataset(df, class_map = fmap, split = 'Train', transform = transform_train, classes = in_classes)     
        test_set = Food101Dataset(df, class_map = fmap, split = 'Test', transform = transform_test, classes = in_classes)     
        ood_set = Food101Dataset(df, class_map = fmap, split = 'T', transform = transform_test, classes = out_classes)
    elif P.dataset == 'car':
        train_set = StanfordCarsDataset(df, class_map = fmap, split = 'Train', transform = transform_train, classes = in_classes)     
        test_set = StanfordCarsDataset(df, class_map = fmap, split = 'Test', transform = transform_test, classes = in_classes)     
        ood_set = StanfordCarsDataset(df, class_map = fmap, split = 'T', transform = transform_test, classes = out_classes) 
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
    )
    ood_loader = DataLoader(
        ood_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, test_loader, ood_loader, size, len(in_classes), transform_test

### Set torch device ###

P.n_gpus = torch.cuda.device_count()
assert P.n_gpus <= 1  # no multi GPU
P.multi_gpu = False

if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

### Initialize dataset ###
ood_eval = P.mode == 'ood_pre'
if P.dataset in ['face', 'car', 'food'] and ood_eval:
    P.batch_size = 1
    P.test_batch_size = 1

df = pd.read_csv('../../datasets/FER2013/icml_face_data.csv')
train_loader, test_loader, ood_loader, image_size, n_classes, transform_test = get_dataloader(df, batch_size = P.batch_size)
#train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset, eval=ood_eval)

P.image_size = image_size
P.n_classes = n_classes

if P.one_class_idx is not None:
    cls_list = get_superclass_list(P.dataset)
    P.n_superclasses = len(cls_list)

    #full_test_set = deepcopy(test_set)  # test set of full classes
    #train_set = get_subclass_dataset(train_set, classes=cls_list[P.one_class_idx])
    #test_set = get_subclass_dataset(test_set, classes=cls_list[P.one_class_idx])

kwargs = {'pin_memory': False, 'num_workers': 4}

#train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
#test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)

if P.ood_dataset is None:
    if P.one_class_idx is not None:
        P.ood_dataset = list(range(P.n_superclasses))
        P.ood_dataset.pop(P.one_class_idx)
    elif P.dataset == 'cifar10':
        P.ood_dataset = ['svhn', 'lsun_resize', 'imagenet_resize', 'lsun_fix', 'imagenet_fix', 'cifar100', 'interp']
    elif P.dataset == 'imagenet':
        P.ood_dataset = ['cub', 'stanford_dogs', 'flowers102', 'places365', 'food_101', 'caltech_256', 'dtd', 'pets']

ood_test_loader = dict()
'''
for ood in P.ood_dataset:
    if ood == 'interp':
        ood_test_loader[ood] = None  # dummy loader
        continue

    if P.one_class_idx is not None:
        #ood_test_set = get_subclass_dataset(full_test_set, classes=cls_list[ood])
        ood_test_set = FaceDataset(df, split = 'T', transform = transform_test, classes = [ood])
        ood = f'one_class_{ood}'  # change save name
    else:
        ood_test_set = get_dataset(P, dataset=ood, test_only=True, image_size=P.image_size, eval=ood_eval)

    ood_test_loader[ood] = DataLoader(ood_test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
'''
for ood in P.ood_dataset:
    ood_test_loader[ood] = ood_loader

### Initialize model ###

simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)
P.shift_trans, P.K_shift = C.get_shift_module(P, eval=True)
P.shift_trans = P.shift_trans.to(device)

model = C.get_classifier(P.model, n_classes=P.n_classes).to(device)
model = C.get_shift_classifer(model, P.K_shift).to(device)
criterion = nn.CrossEntropyLoss().to(device)

if P.load_path is not None:
    checkpoint = torch.load(P.load_path)
    model.load_state_dict(checkpoint, strict=not P.no_strict)
