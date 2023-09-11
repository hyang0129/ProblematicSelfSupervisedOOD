import os
import sys
import numpy as np
import math
import time
import shutil, errno
from distutils.dir_util import copy_tree
import sklearn.metrics as skm
from sklearn.covariance import ledoit_wolf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from tqdm.autonotebook import tqdm

#### logging ####
def save_checkpoint(state, is_best, results_dir, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(results_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(results_dir, filename),
            os.path.join(results_dir, "model_best.pth.tar"),
        )

#### evaluation ####
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def knn(model, device, val_loader):
    """
    Evaluating knn accuracy in feature space.
    Calculates only top-1 accuracy (returns 0 for top-5)
    """

    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1]

            # compute output
            output = F.normalize(model(images), dim=-1).data.cpu()
            features.append(output)
            labels.append(target)

        features = torch.cat(features).numpy()
        labels = torch.cat(labels).numpy()

        cls = KNeighborsClassifier(20, metric="cosine").fit(features, labels)
        acc = 100 * np.mean(cross_val_score(cls, features, labels))

        print(f"knn accuracy for test data = {acc}")

    return acc, 0


def evaluate_acc(model, device, val_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        end = time.time()

        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1]

            pred = model(images)

            pred = torch.argmax(pred, axis=-1)

            correct += (pred.cpu() == target.cpu()).sum().float().cpu()
            total += float(len(pred))

        acc = 100 * (correct / total)

        print(f"softmax accuracy for test data = {acc}")

    return acc, 0

#### OOD detection ####
def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc


def get_pr_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr = skm.average_precision_score(labels, data)
    return aupr


def get_fpr(xin, xood):
    return np.sum(xood < np.percentile(xin, 95)) / len(xood)

def get_features(model, dataloader, device, max_images=10 ** 10, verbose=False):
    features, labels = [], []
    total = 0

    model.eval()

    for index, (img, label) in tqdm(enumerate(dataloader)):

        if total > max_images:
            break

        img, label = img.to(device), label.to(device)

        features += list(model(img).data.cpu().numpy())
        labels += list(label.data.cpu().numpy())

        if verbose and not index % 50:
            print(index)

        total += len(img)

    return np.array(features), np.array(labels)


def msp(x):
    # returns the maximum softmax probability

    return 1 - torch.max(torch.nn.Softmax(dim=-1)(x), dim=-1)[0]


def predict_loop(model, val_loader, device):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        end = time.time()

        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1]

            pred = model(images)

            preds.append(pred)
            targets.append(target)

    preds = torch.concat(preds, axis=0).cpu()
    targets = torch.concat(targets, axis=0).cpu()

    return preds, targets