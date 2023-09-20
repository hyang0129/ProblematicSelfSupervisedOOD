
import faiss
import numpy as np
from benchmark.utils import (
    get_roc_sklearn,
    get_pr_sklearn,
    get_fpr,
    get_features,
    msp,
    predict_loop
)
import torch
from loguru import logger
import cupy as cp
from tqdm.autonotebook import tqdm

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


    return fpr95, auroc, aupr, dtest, dood

def get_eval_results_msp(ftest, food):
    """
    None.
    """
    # standardize data

    dtest = msp(ftest).numpy()
    dood = msp(food).numpy()

    fpr95 = get_fpr(dtest, dood)
    auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
    return fpr95, auroc, aupr, dtest, dood





def get_eval_results_knn(ftrain, ftest, food, args):


    ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
    ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
    food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10

    m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)

    ftrain = (ftrain - m) / (s + 1e-10)
    ftest = (ftest - m) / (s + 1e-10)
    food = (food - m) / (s + 1e-10)

    cpftrain = cp.array(ftrain)

    def calculate_nearest_neighbor(feature, cpftrain=cpftrain , k=50):
        diff = cpftrain - feature

        diff = cp.linalg.norm(diff, axis=-1)
        a = diff

        return cp.sum(a[cp.argsort(a)[k]])

    f = calculate_nearest_neighbor
    test_scores = []
    n = 1000
    iters = int(len(ftest)/n) + 1
    for i in tqdm(range(iters)):
        test_scores.append(cp.apply_along_axis(f, 1, cp.array(ftest[i * n:(1 + i) * n])).get())

    iters = int(len(food) / n) + 1
    ood_scores = []
    for i in tqdm(range(iters)):
        ood_scores.append(cp.apply_along_axis(f, 1, cp.array(food[i * n:(1 + i) * n])).get())

    dtest = np.concatenate(test_scores)
    dood = np.concatenate(ood_scores)

    fpr95 = get_fpr(dtest, dood)
    auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)

    return fpr95, auroc, aupr, dtest, dood

def run_evaluation(model,
                   args,
                   device,
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
            model.encoder, train_loader, device
        )

        features_test, _ = get_features(model.encoder, test_loader, device=device)
        logger.info("In-distribution features shape: ", features_train.shape, features_test.shape)

        features_ood, _ = get_features(model.encoder, ood_loader, device=device)

        fpr95, auroc, aupr, dtest, dood = get_eval_results_clustering(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
            args,
        )

    elif args.training_mode == 'SupCon':

        test_pred, _ = predict_loop(model, test_loader, device)
        ood_pred, _ = predict_loop(model, ood_loader, device)

        fpr95, auroc, aupr, dtest, dood = get_eval_results_msp(test_pred, ood_pred)


    else:
        raise KeyError(f'Training Mode {args.training_mode} not recognized')

    logger.info(f'FPR95 = {fpr95}, AUROC = {auroc}, AUPR = {aupr}')

    return fpr95, auroc, aupr, dtest, dood