import tensorflow as tf
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
from PIL import Image



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

    filterd_ds = filterd_ds.filter(lambda x: notfun(tf.reduce_any(x['label'] == tfindist)))

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