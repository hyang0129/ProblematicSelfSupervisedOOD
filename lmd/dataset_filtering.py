import tensorflow as tf
import pandas as pd
from tqdm.notebook import tqdm



def get_indist_classes(eval_ds, random_state = 0):
    labels = []

    for data in eval_ds:
        labels.append(data['label'])

    all_unique_labels = sorted(list(tf.unique(tf.concat(labels, axis=0))[0].numpy()))

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
        batch_size = data['label'].shape[0]
        break

    filterd_ds = dataset.unbatch()

    filterd_ds = filterd_ds.filter(lambda x: notfun(tf.reduce_any(x['label'] == tfindist)))
    filterd_ds = filterd_ds.batch(batch_size)

    return filterd_ds