import tensorflow_datasets as tfds
import os
import pandas as pd
from tqdm.autonotebook import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from torchvision import transforms
from benchmark.datasets.affectnet import Affectnet, FaceDataset
import numpy as np


class OODDataset(Dataset):

    def __init__(self, dataset_name, split='Out', transform=None, random_state = 42, rotnet = False):
        '''
        Returns a datasets split based on 75% 25%.

        :param dataset_name:
        :param split:
        :param transform:
        '''


        assert split in ['Train', 'Test', 'Out']

        self.rotnet = rotnet
        self.dataset_name = dataset_name
        self.return_grouped_cifar = False

        datasource, dataframe = self.build(dataset_name)

        self.dataframe = dataframe
        df = dataframe

        self.in_distro = pd.Series(df.label.unique()).sample(frac=0.75, random_state=random_state).values

        self.out_distro = pd.Series(df.label.unique())[~pd.Series(df.label.unique()).isin(self.in_distro)].values

        new_label_int = {old: new_label_int for new_label_int, old in enumerate(self.in_distro)} # labels 1 5 6 9 to 0 1 2 3
        new_label_int = None if self.return_grouped_cifar else new_label_int

        if split == 'Train':
            self.df = df[(df.split == 'train') & (df.label.isin(self.in_distro))].reset_index()
            self.source = datasource['train']
        elif split == 'Test':
            self.df = df[(df.split == 'val') & (df.label.isin(self.in_distro))].reset_index()
            self.source = datasource['validation']
        elif split == 'Out':
            self.df = df[(df.split == 'val') & (~df.label.isin(self.in_distro))].reset_index()
            self.source = datasource['validation']

        self.transform = transform
        self.new_label_int = None if split == 'Out' else new_label_int


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        label = row['label']
        if self.new_label_int is not None:
            label = self.new_label_int[label]
        elif self.return_grouped_cifar:
            label = row['group_idx']

        data = self.source[int(row['index'])]

        if type(data) is dict: # TFDS dict style
            image = data['image']
        else:
            image = data[0] # Pytorch tuple style

        if type(image) is not Image.Image:
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        if self.rotnet:
            rots = torch.randint(0, high = 4, size =  ())
            image = torchvision.transforms.functional.rotate(image, angle = int(90 * rots))
            label = rots

        return image, label

    def build(self, dataset_name):

        if dataset_name == 'food101':
            datasource = tfds.data_source('food101')

            if not os.path.exists('food_label_index.csv'):
                label_info = []
                for i, data in tqdm(enumerate(datasource['train'])):
                    label_info.append({'split': 'train', 'index': i, 'label': data['label']})

                for i, data in tqdm(enumerate(datasource['validation'])):
                    label_info.append({'split': 'val', 'index': i, 'label': data['label']})

                    df = pd.DataFrame(label_info)
                    df.to_csv('food_label_index.csv')

                dataframe = df
            else:
                dataframe = pd.read_csv('food_label_index.csv', index_col=[0])

        elif dataset_name == 'cars':

            train_data = torchvision.datasets.StanfordCars(root="data", download=True, split='train')
            test_data = torchvision.datasets.StanfordCars(root="data", download=True, split='test')

            label_file = 'cars_label_index.csv'

            ds = {'train': train_data, 'validation': test_data}
            datasource = {'train': train_data, 'validation': test_data}

            if not os.path.exists(label_file):
                label_info = []

                for i, data in tqdm(enumerate(ds['train'])):
                    label_info.append({'split': 'train', 'index': i, 'label': data[1]})

                for i, data in tqdm(enumerate(ds['validation'])):
                    label_info.append({'split': 'val', 'index': i, 'label': data[1]})

                df = pd.DataFrame(label_info)
                df.to_csv(label_file)
            else:
                df = pd.read_csv(label_file, index_col=[0])

            dataframe = df

        elif dataset_name == 'affectnet':
            df = pd.read_csv('affectnet_label_index.csv', index_col=[0])
            df[df.index < 100000]
            train_data = Affectnet(df, split='train')
            test_data = Affectnet(df, split='val')

            datasource = {'train': train_data, 'validation': test_data}

            dataframe = df

        elif dataset_name == 'icmlface':
            df = pd.read_csv('data/icml_face_data.csv')

            df['split'] = 'unknown'

            df.loc[df[' Usage'] == 'Training', 'split'] = 'train'
            df.loc[df[' Usage'].str.contains('Test'), 'split'] = 'val'
            df['label'] = df['emotion']
            df['index'] = df.index
            dataset = FaceDataset(df)
            datasource = {'train': dataset, 'validation': dataset}
            dataframe = df

        elif dataset_name == 'icmlface':
            df = pd.read_csv('data/icml_face_data.csv')

            df['split'] = 'unknown'

            df.loc[df[' Usage'] == 'Training', 'split'] = 'train'
            df.loc[df[' Usage'].str.contains('Test'), 'split'] = 'val'
            df['label'] = df['emotion']
            df['index'] = df.index

            dataset = FaceDataset(df)

            datasource = {'train': dataset, 'validation': dataset}

            dataframe = df


        elif dataset_name in ['cifar10']:


            label_file = 'cifar10_index.csv'

            train_data = torchvision.datasets.CIFAR10(root='.', download=True, train=True)
            test_data = torchvision.datasets.CIFAR10(root='.', download=True, train=False)

            ds = {'train': train_data, 'validation': test_data}
            datasource = {'train': train_data, 'validation': test_data}

            if not os.path.exists(label_file):
                label_info = []

                for i, data in tqdm(enumerate(ds['train'])):
                    label_info.append({'split': 'train', 'index': i, 'label': data[1]})

                for i, data in tqdm(enumerate(ds['validation'])):
                    label_info.append({'split': 'val', 'index': i, 'label': data[1]})

                df = pd.DataFrame(label_info)
                df.to_csv(label_file)
            else:
                df = pd.read_csv(label_file, index_col=[0])

            dataframe = df



        elif dataset_name in ['cifar100']:

            label_file = 'cifar100_index.csv'

            train_data = torchvision.datasets.CIFAR100(root='.', download=True, train=True)
            test_data = torchvision.datasets.CIFAR100(root='.', download=True, train=False)

            ds = {'train': train_data, 'validation': test_data}
            datasource = {'train': train_data, 'validation': test_data}

            if not os.path.exists(label_file):
                label_info = []

                for i, data in tqdm(enumerate(ds['train'])):
                    label_info.append({'split': 'train', 'index': i, 'label': data[1]})

                for i, data in tqdm(enumerate(ds['validation'])):
                    label_info.append({'split': 'val', 'index': i, 'label': data[1]})

                df = pd.DataFrame(label_info)
                df.to_csv(label_file)
            else:
                df = pd.read_csv(label_file, index_col=[0])

            dataframe = df


        else:
            raise KeyError(f'Dataset name {dataset_name} is not valid')

        return datasource, dataframe

class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def get_dataloaders(dataset_name, args, batch_size=32, normalize=True, size=32, doCLR = False, random_state = 42, num_workers = 16):

    transform_train = [
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]

    if args.training_mode == "RotNet":
        transform_train = [
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]

    transform_test = [transforms.Resize([size, size]), transforms.ToTensor()]

    norm_layer = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
    )

    if normalize:
        transform_train.append(norm_layer)
        transform_test.append(norm_layer)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    if args.training_mode == "SimCLR" and doCLR:
        transform_train = TwoCropTransform(transform_train)

    train_set = OODDataset(dataset_name, split='Train', transform=transform_train, rotnet=args.training_mode == "RotNet", random_state = random_state)
    test_set = OODDataset(dataset_name, split='Test', transform=transform_test, random_state = random_state)
    ood_set = OODDataset(dataset_name, split='Out', transform=transform_test, random_state = random_state)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    ood_loader = DataLoader(
        ood_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader, ood_loader, train_set, test_set, ood_set

def get_inv_norm():
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=1 / np.array([0.2675, 0.2565, 0.2761])),
                                   transforms.Normalize(mean=np.array([0.5071, 0.4867, 0.4408]) * -1,
                                                        std=[1., 1., 1.]),
                                   ])

    return invTrans