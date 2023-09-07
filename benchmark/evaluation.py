
import faiss
import numpy as np
from utils import (
    get_roc_sklearn,
    get_pr_sklearn,
    get_fpr,
    get_features,
    msp,
)
import torch

def get_scores(ftrain, ftest, food, labelstrain, args):
    if args.clusters == 1:
        return get_scores_one_cluster(ftrain, ftest, food)


def get_clusters(ftrain, nclusters):
    kmeans = faiss.Kmeans(
        ftrain.shape[1], nclusters, niter=100, verbose=False, gpu=False
    )
    kmeans.train(np.random.permutation(ftrain))
    _, ypred = kmeans.assign(ftrain)
    return ypred

def get_scores_one_cluster(ftrain, ftest, food):

    cov = lambda x: np.cov(x.T, bias=True)

    dtest = np.sum(
        (ftest - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (ftest - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    dood = np.sum(
        (food - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (food - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    return dtest, dood


def get_eval_results_clustering(ftrain, ftest, food, labelstrain, args):
    """
    None.
    """
    # standardize data
    ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
    ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
    food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10

    m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)

    ftrain = (ftrain - m) / (s + 1e-10)
    ftest = (ftest - m) / (s + 1e-10)
    food = (food - m) / (s + 1e-10)

    dtest, dood = get_scores(ftrain, ftest, food, labelstrain, args)

    fpr95 = get_fpr(dtest, dood)
    auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
    return fpr95, auroc, aupr

def get_eval_results_msp(ftest, food):
    """
    None.
    """
    # standardize data

    dtest = msp(ftest).numpy()
    dood = msp(food).numpy()

    fpr95 = get_fpr(dtest, dood)
    auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
    return fpr95, auroc, aupr

def run_evaluation(model,
                   args,
                   checkpoint_path,
                   train_loader,
                   test_loader,
                   ood_loader,
                   ):

    ckpt_dict = torch.load(checkpoint_path)
    args.clusters = 1
    model.load_state_dict(ckpt_dict['state_dict'] if 'state_dict' in ckpt_dict.keys() else ckpt_dict['model'])

    if args.training_mode == 'SimCLR':
        features_train, labels_train = get_features(
            model.encoder, train_loader
        )

        features_test, _ = get_features(model.encoder, test_loader)
        print("In-distribution features shape: ", features_train.shape, features_test.shape)

        features_ood, _ = get_features(model.encoder, ood_loader)

        fpr95, auroc, aupr = get_eval_results_clustering(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
            args,
        )

    else:
        raise KeyError(f'Training Mode {args.training_mode} not recognized')

    return fpr95, auroc, aupr