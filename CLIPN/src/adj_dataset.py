import tensorflow as tf
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds
import os
from tensorflow_datasets.image_classification.cars196 import Cars196
import torchvision.transforms as T
import torch


def get_indist_classes(eval_ds, random_state = 0):
    labels = []

    for data in eval_ds:
        labels.append(data['label'])

    all_unique_labels = sorted(list(tf.unique(tf.reshape(tf.stack(labels), (-1,)))[0].numpy()))

    all_unique_labels = pd.Series(all_unique_labels)

    indist = all_unique_labels.sample(frac=0.75, random_state=random_state).values


    tfindist = tf.convert_to_tensor(indist, dtype=tf.int64)

    # oodist = all_unique_labels[~all_unique_labels.isin(indist)].values

    return tfindist


# %%
def apply_class_filter(dataset, tfindist, reverse=False):
    if reverse:
        notfun = tf.math.logical_not
    else:
        notfun = lambda x: x

    for data in dataset:
        break


    filterd_ds =  dataset
    if len(data['label'].shape) > 0:
        batch_size = data['label'].shape[0]
        filterd_ds = filterd_ds.unbatch()

    filterd_ds = filterd_ds.filter(lambda x: notfun(tf.reduce_any( tf.cast(x['label'], dtype= tf.int64) == tf.cast(tfindist, dtype= tf.int64)  )))

    if len(data['label'].shape) > 0:
        filterd_ds = filterd_ds.batch(batch_size)

    return filterd_ds



def get_icml_face(df_path='data/icml_face_data.csv'):
    '''
    Creates the ICML face dataset as a tf.data.Dataset

    We need a path to the icml face csv file which can be downloaded from here:  https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=icml_face_data.csv.

    :param df_path:
    :return:
    '''

    def convert_to_image(row):
        label = row["emotion"]
        pixels = row[" pixels"]

        pixels = tf.strings.to_number(tf.strings.split(pixels, sep=' '), out_type=tf.int32)
        pixels = tf.reshape(pixels, (48, 48, 1))
        pixels = tf.repeat(pixels, repeats=3, axis=-1)
        image = tf.cast(pixels, tf.uint8)

        if row[" Usage"] == "Training":
            split = 'train'
        else:
            split = 'test'

        return {'image': image, 'label': label, 'split': split}


    df = pd.read_csv(df_path)

    ds = tf.data.Dataset.from_tensor_slices(df.to_dict("list"))
    ds = ds.map(convert_to_image)

    train_ds = ds.filter(lambda x: x['split'] == 'train')
    test_ds = ds.filter(lambda x: x['split'] == 'test')

    return train_ds, test_ds

class Cars196_fixed(Cars196):
    '''
    A fixed version of the Cars196 dataset, where we follow the same manual download instructions as Cars196
    for pytorch. We simply set the prefix to reference the stanford_cars folder used in the pytorch dataset.
    see https://github.com/pytorch/vision/issues/7545
    '''


    def _split_generators(self, dl_manager):
        prefix = './data/stanford_cars'

        output_files = {
            'train': prefix,
            'test': prefix,
            'extra': prefix,
            'test_annos': os.path.join(prefix, 'cars_test_annos_withlabels.mat')
        }

        return [
            tfds.core.SplitGenerator(
                name='train',
                gen_kwargs={
                    'split_name': 'train',
                    'data_dir_path': os.path.join(
                        output_files['train'], 'cars_train'
                    ),
                    'data_annotations_path': os.path.join(
                        output_files['extra'],
                        os.path.join('devkit', 'cars_train_annos.mat'),
                    ),
                },
            ),
            tfds.core.SplitGenerator(
                name='test',
                gen_kwargs={
                    'split_name': 'test',
                    'data_dir_path': os.path.join(
                        output_files['test'], 'cars_test'
                    ),
                    'data_annotations_path': output_files['test_annos'],
                },
            ),
        ]

class IterableImageDataset(torch.utils.data.IterableDataset):
  def __init__(self, tf_dataset, transforms = None):
    self.tf_dataset = tf_dataset
    self.transforms = transforms

  def __len__(self):
    return len(self.tf_dataset)

  def apply_fn(self, data):

    img = Image.fromarray(data['image'].numpy())

    # dataset should already be RGB
    # if img.mode != "RGB":
    #   img = img.convert("RGB")

    # img = T.ToTensor()(img)

    if self.transform is not None:
        img = self.transform(img)

    # note that this is intended for zero shot OOD so the class is not relevant
    return img, 0

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:  # single-process data loading, return the full iterator
      pass
    else:  # in a worker process
      raise NotImplementedError('Not implemented for more than 1 worker')
    return map(self.apply_fn, self.tf_dataset)


class FaceDataset:
    def __init__(self, preprocess_train=None, preprocess_test=None,
                 batch_size=128,
                 num_workers=0,
                 classnames=None,
                 seed = 0):

        tf_train, tf_test = get_icml_face(df_path='data/icml_face_data.csv')
        # in order from 0 to 6
        self.classnames = ["angry", "disgust", 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.classes = self.classnames

        indist = get_indist_classes(tf_test, random_state=seed)

        self.classes = [self.classnames[i] for i in indist]
        self.classnames = self.classes

        print(f'ID CLASSES :{self.classes}')

        self.train_id_dataset = IterableImageDataset(apply_class_filter(tf_train, indist, reverse=False), transforms=preprocess_train)
        self.test_id_dataset = IterableImageDataset(apply_class_filter(tf_test, indist, reverse=False), transforms=preprocess_test)
        self.test_ood_dataset = IterableImageDataset(apply_class_filter(tf_test, indist, reverse=True), transforms=preprocess_test)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_id_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_id_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.ood_loader = torch.utils.data.DataLoader(
            self.test_ood_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )


# from tfds
cars_names = [
    'AM General Hummer SUV 2000',
    'Acura RL Sedan 2012',
    'Acura TL Sedan 2012',
    'Acura TL Type-S 2008',
    'Acura TSX Sedan 2012',
    'Acura Integra Type R 2001',
    'Acura ZDX Hatchback 2012',
    'Aston Martin V8 Vantage Convertible 2012',
    'Aston Martin V8 Vantage Coupe 2012',
    'Aston Martin Virage Convertible 2012',
    'Aston Martin Virage Coupe 2012',
    'Audi RS 4 Convertible 2008',
    'Audi A5 Coupe 2012',
    'Audi TTS Coupe 2012',
    'Audi R8 Coupe 2012',
    'Audi V8 Sedan 1994',
    'Audi 100 Sedan 1994',
    'Audi 100 Wagon 1994',
    'Audi TT Hatchback 2011',
    'Audi S6 Sedan 2011',
    'Audi S5 Convertible 2012',
    'Audi S5 Coupe 2012',
    'Audi S4 Sedan 2012',
    'Audi S4 Sedan 2007',
    'Audi TT RS Coupe 2012',
    'BMW ActiveHybrid 5 Sedan 2012',
    'BMW 1 Series Convertible 2012',
    'BMW 1 Series Coupe 2012',
    'BMW 3 Series Sedan 2012',
    'BMW 3 Series Wagon 2012',
    'BMW 6 Series Convertible 2007',
    'BMW X5 SUV 2007',
    'BMW X6 SUV 2012',
    'BMW M3 Coupe 2012',
    'BMW M5 Sedan 2010',
    'BMW M6 Convertible 2010',
    'BMW X3 SUV 2012',
    'BMW Z4 Convertible 2012',
    'Bentley Continental Supersports Conv. Convertible 2012',
    'Bentley Arnage Sedan 2009',
    'Bentley Mulsanne Sedan 2011',
    'Bentley Continental GT Coupe 2012',
    'Bentley Continental GT Coupe 2007',
    'Bentley Continental Flying Spur Sedan 2007',
    'Bugatti Veyron 16.4 Convertible 2009',
    'Bugatti Veyron 16.4 Coupe 2009',
    'Buick Regal GS 2012',
    'Buick Rainier SUV 2007',
    'Buick Verano Sedan 2012',
    'Buick Enclave SUV 2012',
    'Cadillac CTS-V Sedan 2012',
    'Cadillac SRX SUV 2012',
    'Cadillac Escalade EXT Crew Cab 2007',
    'Chevrolet Silverado 1500 Hybrid Crew Cab 2012',
    'Chevrolet Corvette Convertible 2012',
    'Chevrolet Corvette ZR1 2012',
    'Chevrolet Corvette Ron Fellows Edition Z06 2007',
    'Chevrolet Traverse SUV 2012',
    'Chevrolet Camaro Convertible 2012',
    'Chevrolet HHR SS 2010',
    'Chevrolet Impala Sedan 2007',
    'Chevrolet Tahoe Hybrid SUV 2012',
    'Chevrolet Sonic Sedan 2012',
    'Chevrolet Express Cargo Van 2007',
    'Chevrolet Avalanche Crew Cab 2012',
    'Chevrolet Cobalt SS 2010',
    'Chevrolet Malibu Hybrid Sedan 2010',
    'Chevrolet TrailBlazer SS 2009',
    'Chevrolet Silverado 2500HD Regular Cab 2012',
    'Chevrolet Silverado 1500 Classic Extended Cab 2007',
    'Chevrolet Express Van 2007',
    'Chevrolet Monte Carlo Coupe 2007',
    'Chevrolet Malibu Sedan 2007',
    'Chevrolet Silverado 1500 Extended Cab 2012',
    'Chevrolet Silverado 1500 Regular Cab 2012',
    'Chrysler Aspen SUV 2009',
    'Chrysler Sebring Convertible 2010',
    'Chrysler Town and Country Minivan 2012',
    'Chrysler 300 SRT-8 2010',
    'Chrysler Crossfire Convertible 2008',
    'Chrysler PT Cruiser Convertible 2008',
    'Daewoo Nubira Wagon 2002',
    'Dodge Caliber Wagon 2012',
    'Dodge Caliber Wagon 2007',
    'Dodge Caravan Minivan 1997',
    'Dodge Ram Pickup 3500 Crew Cab 2010',
    'Dodge Ram Pickup 3500 Quad Cab 2009',
    'Dodge Sprinter Cargo Van 2009',
    'Dodge Journey SUV 2012',
    'Dodge Dakota Crew Cab 2010',
    'Dodge Dakota Club Cab 2007',
    'Dodge Magnum Wagon 2008',
    'Dodge Challenger SRT8 2011',
    'Dodge Durango SUV 2012',
    'Dodge Durango SUV 2007',
    'Dodge Charger Sedan 2012',
    'Dodge Charger SRT-8 2009',
    'Eagle Talon Hatchback 1998',
    'FIAT 500 Abarth 2012',
    'FIAT 500 Convertible 2012',
    'Ferrari FF Coupe 2012',
    'Ferrari California Convertible 2012',
    'Ferrari 458 Italia Convertible 2012',
    'Ferrari 458 Italia Coupe 2012',
    'Fisker Karma Sedan 2012',
    'Ford F-450 Super Duty Crew Cab 2012',
    'Ford Mustang Convertible 2007',
    'Ford Freestar Minivan 2007',
    'Ford Expedition EL SUV 2009',
    'Ford Edge SUV 2012',
    'Ford Ranger SuperCab 2011',
    'Ford GT Coupe 2006',
    'Ford F-150 Regular Cab 2012',
    'Ford F-150 Regular Cab 2007',
    'Ford Focus Sedan 2007',
    'Ford E-Series Wagon Van 2012',
    'Ford Fiesta Sedan 2012',
    'GMC Terrain SUV 2012',
    'GMC Savana Van 2012',
    'GMC Yukon Hybrid SUV 2012',
    'GMC Acadia SUV 2012',
    'GMC Canyon Extended Cab 2012',
    'Geo Metro Convertible 1993',
    'HUMMER H3T Crew Cab 2010',
    'HUMMER H2 SUT Crew Cab 2009',
    'Honda Odyssey Minivan 2012',
    'Honda Odyssey Minivan 2007',
    'Honda Accord Coupe 2012',
    'Honda Accord Sedan 2012',
    'Hyundai Veloster Hatchback 2012',
    'Hyundai Santa Fe SUV 2012',
    'Hyundai Tucson SUV 2012',
    'Hyundai Veracruz SUV 2012',
    'Hyundai Sonata Hybrid Sedan 2012',
    'Hyundai Elantra Sedan 2007',
    'Hyundai Accent Sedan 2012',
    'Hyundai Genesis Sedan 2012',
    'Hyundai Sonata Sedan 2012',
    'Hyundai Elantra Touring Hatchback 2012',
    'Hyundai Azera Sedan 2012',
    'Infiniti G Coupe IPL 2012',
    'Infiniti QX56 SUV 2011',
    'Isuzu Ascender SUV 2008',
    'Jaguar XK XKR 2012',
    'Jeep Patriot SUV 2012',
    'Jeep Wrangler SUV 2012',
    'Jeep Liberty SUV 2012',
    'Jeep Grand Cherokee SUV 2012',
    'Jeep Compass SUV 2012',
    'Lamborghini Reventon Coupe 2008',
    'Lamborghini Aventador Coupe 2012',
    'Lamborghini Gallardo LP 570-4 Superleggera 2012',
    'Lamborghini Diablo Coupe 2001',
    'Land Rover Range Rover SUV 2012',
    'Land Rover LR2 SUV 2012',
    'Lincoln Town Car Sedan 2011',
    'MINI Cooper Roadster Convertible 2012',
    'Maybach Landaulet Convertible 2012',
    'Mazda Tribute SUV 2011',
    'McLaren MP4-12C Coupe 2012',
    'Mercedes-Benz 300-Class Convertible 1993',
    'Mercedes-Benz C-Class Sedan 2012',
    'Mercedes-Benz SL-Class Coupe 2009',
    'Mercedes-Benz E-Class Sedan 2012',
    'Mercedes-Benz S-Class Sedan 2012',
    'Mercedes-Benz Sprinter Van 2012',
    'Mitsubishi Lancer Sedan 2012',
    'Nissan Leaf Hatchback 2012',
    'Nissan NV Passenger Van 2012',
    'Nissan Juke Hatchback 2012',
    'Nissan 240SX Coupe 1998',
    'Plymouth Neon Coupe 1999',
    'Porsche Panamera Sedan 2012',
    'Ram C/V Cargo Van Minivan 2012',
    'Rolls-Royce Phantom Drophead Coupe Convertible 2012',
    'Rolls-Royce Ghost Sedan 2012',
    'Rolls-Royce Phantom Sedan 2012',
    'Scion xD Hatchback 2012',
    'Spyker C8 Convertible 2009',
    'Spyker C8 Coupe 2009',
    'Suzuki Aerio Sedan 2007',
    'Suzuki Kizashi Sedan 2012',
    'Suzuki SX4 Hatchback 2012',
    'Suzuki SX4 Sedan 2012',
    'Tesla Model S Sedan 2012',
    'Toyota Sequoia SUV 2012',
    'Toyota Camry Sedan 2012',
    'Toyota Corolla Sedan 2012',
    'Toyota 4Runner SUV 2012',
    'Volkswagen Golf Hatchback 2012',
    'Volkswagen Golf Hatchback 1991',
    'Volkswagen Beetle Hatchback 2012',
    'Volvo C30 Hatchback 2012',
    'Volvo 240 Sedan 1993',
    'Volvo XC90 SUV 2007',
    'smart fortwo Convertible 2012',
]

class CarsDataset:
    def __init__(self, preprocess_train=None, preprocess_test=None,
                 batch_size=128,
                 num_workers=0,
                 classnames=None,
                 seed=0):

        self.classnames = cars_names
        self.classes = self.classnames

        dataset_builder = Cars196_fixed()

        dataset_builder.download_and_prepare()
        tf_train = dataset_builder.as_dataset(
        split='train', shuffle_files=True)

        tf_test = dataset_builder.as_dataset(
        split='test', shuffle_files=True)

        indist = get_indist_classes(tf_test, random_state=seed)

        self.classes = [self.classnames[i] for i in indist]
        self.classnames = self.classes

        print(f'ID CLASSES :{self.classes}')

        self.train_id_dataset = IterableImageDataset(apply_class_filter(tf_train, tf_test, reverse=False), transforms=preprocess_train)
        self.test_id_dataset = IterableImageDataset(apply_class_filter(tf_test, tf_test, reverse=False), transforms=preprocess_test)
        self.test_ood_dataset = IterableImageDataset(apply_class_filter(tf_test, tf_test, reverse=True), transforms=preprocess_test)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_id_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_id_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.ood_loader = torch.utils.data.DataLoader(
            self.test_ood_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )



class FoodDataset:
    def __init__(self, preprocess_train=None, preprocess_test=None,
                 batch_size=128,
                 num_workers=0,
                 classnames=None,
                 seed=0):
        pass

class Cifar10Dataset:
    def __init__(self, preprocess_train=None, preprocess_test=None,
                 batch_size=128,
                 num_workers=0,
                 classnames=None,
                 seed=0):
        pass



class Cifar100Dataset:
    def __init__(self, preprocess_train=None, preprocess_test=None,
                 batch_size=128,
                 num_workers=0,
                 classnames=None,
                 seed=0):
        pass