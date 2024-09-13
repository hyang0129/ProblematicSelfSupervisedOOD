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

    filterd_ds = filterd_ds.filter(lambda x: notfun(tf.reduce_any( tf.cast(x['label'], dtype= tf.int64) == tfindist)))

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


class CarsDataset:
    def __init__(self, preprocess_train=None, preprocess_test=None,
                 batch_size=128,
                 num_workers=0,
                 classnames=None,
                 seed=0):
        pass

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