from argparse import ArgumentParser
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import models.classifier as C
from datasets import get_dataset, get_superclass_list, get_subclass_dataset
from utils.utils import load_checkpoint

from torch.utils.data import Dataset, DataLoader
from custom_datasets import get_dataloader
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

from PIL import Image
import sys 
import os
from pathlib import Path
import cv2

from utils.utils import Logger
from utils.utils import save_checkpoint
from utils.utils import save_linear_checkpoint

from evals import test_classifier

def evaluate(P, run, ood_layer_eval):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    P.resize_factor = 0.54
    P.resize_fix  = True
    P.ood_layer = ood_layer_eval

    ood_eval = P.ood_mode == 'ood_pre'
    if P.dataset in ['face', 'car', 'food'] and ood_eval:
        P.batch_size = 1
        P.test_batch_size = 1
        
    train_loader, test_loader, ood_loader, image_size, n_classes, transform_test = get_dataloader(P, evaluate = True, batch_size = P.batch_size)

    simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)
    P.shift_trans, P.K_shift = C.get_shift_module(P, eval=True)
    P.shift_trans = P.shift_trans.to(device)

    model = C.get_classifier(P.model, n_classes=P.n_classes).to(device)
    model = C.get_shift_classifer(model, P.K_shift).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    checkpoint = torch.load(os.path.join(P.load_path, str(run), 'last.model'))
    model.load_state_dict(checkpoint, strict=not P.no_strict)

    model.eval()
    ood_test_loader = dict()
    for ood in P.ood_dataset:
        ood_test_loader[ood] = ood_loader

    if P.ood_mode == 'test_acc':
        from evals import test_classifier
        with torch.no_grad():
            error = test_classifier(P, model, test_loader, 0, logger=None)

    elif P.ood_mode == 'test_marginalized_acc':
        from evals import test_classifier
        with torch.no_grad():
            error = test_classifier(P, model, test_loader, 0, marginal=True, logger=None)

    elif P.ood_mode in ['ood', 'ood_pre']:
        if P.ood_mode == 'ood':
            from evals import eval_ood_detection
        else:
            from evals.ood_pre import eval_ood_detection

        with torch.no_grad():
            auroc_dict, aupr_dict, fpr95_dict = eval_ood_detection(P, model, test_loader, ood_test_loader, P.ood_score,
                                            train_loader=train_loader, simclr_aug=simclr_aug)

        if P.one_class_idx is not None:
            mean_dict = dict()
            for ood_score in P.ood_score:
                mean = 0
                for ood in auroc_dict.keys():
                    mean += auroc_dict[ood][ood_score]
                mean_dict[ood_score] = mean / len(auroc_dict.keys())
            auroc_dict['one_class_mean'] = mean_dict

        return auroc_dict, aupr_dict, fpr95_dict

    else:
        raise NotImplementedError()


def main():
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of CSI')

    parser.add_argument('--dataset', help='Dataset',
                        choices=['cifar10', 'cifar100', 'imagenet', 'face', 'food', 'car'], type=str)
    parser.add_argument('--one_class_idx', help='None: multi-class, Not None: one-class',
                        default=None, type=int)
    parser.add_argument('--model', help='Model',
                        choices=['resnet18', 'resnet18_imagenet'], type=str)
    parser.add_argument('--mode', help='Training mode',
                        default='simclr', type=str)
    parser.add_argument('--simclr_dim', help='Dimension of simclr layer',
                        default=128, type=int)

    parser.add_argument('--shift_trans_type', help='shifting transformation type', default='none',
                        choices=['rotation', 'cutperm', 'none'], type=str)

    parser.add_argument("--local_rank", type=int,
                        default=0, help='Local rank for distributed learning')
    parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)
    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        default=None, type=str)
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')
    parser.add_argument('--suffix', help='Suffix for the log dir',
                        default=None, type=str)
    parser.add_argument('--error_step', help='Epoch steps to compute errors',
                        default=5, type=int)
    parser.add_argument('--save_step', help='Epoch steps to save models',
                        default=10, type=int)

    ##### Training Configurations #####
    parser.add_argument('--runs', help='Number of runs of epochs',
                        default=3, type=int)
    parser.add_argument('--epochs', help='Epochs',
                        default=1000, type=int)
    parser.add_argument('--optimizer', help='Optimizer',
                        choices=['sgd', 'lars'],
                        default='lars', type=str)
    parser.add_argument('--lr_scheduler', help='Learning rate scheduler',
                        choices=['step_decay', 'cosine'],
                        default='cosine', type=str)
    parser.add_argument('--warmup', help='Warm-up epochs',
                        default=10, type=int)
    parser.add_argument('--lr_init', help='Initial learning rate',
                        default=1e-1, type=float)
    parser.add_argument('--weight_decay', help='Weight decay',
                        default=1e-6, type=float)
    parser.add_argument('--batch_size', help='Batch size',
                        default=128, type=int)
    parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=100, type=int)

    ##### Objective Configurations #####
    parser.add_argument('--sim_lambda', help='Weight for SimCLR loss',
                        default=1.0, type=float)
    parser.add_argument('--temperature', help='Temperature for similarity',
                        default=0.5, type=float)

    ##### Evaluation Configurations #####
    parser.add_argument("--ood_dataset", help='Datasets for OOD detection',
                        default=None, nargs="*", type=str)
    parser.add_argument("--ood_score", help='score function for OOD detection',
                        default=['norm_mean'], nargs="+", type=str)
    parser.add_argument("--ood_layer", help='layer for OOD scores',
                        choices=['penultimate', 'simclr', 'shift'],
                        default=['simclr', 'shift'], nargs="+", type=str)
    parser.add_argument("--ood_samples", help='number of samples to compute OOD score',
                        default=1, type=int)
    parser.add_argument("--ood_batch_size", help='batch size to compute OOD score',
                        default=100, type=int)
    parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',
                        default=0.08, type=float)
    parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',
                        action='store_true')
    parser.add_argument('--ood_mode', help='Evaluation mode',
                        default='ood_pre', type=str)

    parser.add_argument("--print_score", help='print quantiles of ood score',
                        action='store_true')
    parser.add_argument("--save_score", help='save ood score for plotting histogram',
                        action='store_true')

    P = parser.parse_args()

    print("Resize fix: ", P.resize_fix)

    ### Set torch device ###

    if torch.cuda.is_available():
        torch.cuda.set_device(P.local_rank)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    P.n_gpus = torch.cuda.device_count()

    if P.n_gpus > 1:
        import apex
        import torch.distributed as dist
        from torch.utils.data.distributed import DistributedSampler

        P.multi_gpu = True
        torch.distributed.init_process_group(
            'nccl',
            init_method='env://',
            world_size=P.n_gpus,
            rank=P.local_rank,
        )
    else:
        P.multi_gpu = False

    ### only use one ood_layer while training
    ood_layer_eval = P.ood_layer
    P.ood_layer = P.ood_layer[0]

    ### Initialize dataset ###
    train_loader, test_loader, ood_loader, image_size, n_classes, _ = get_dataloader(P, batch_size = P.batch_size)
    #train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset)
    P.image_size = image_size
    P.n_classes = n_classes

    if P.one_class_idx is not None:
        cls_list = get_superclass_list(P.dataset)
        P.n_superclasses = len(cls_list)

        #full_test_set = deepcopy(test_set)  # test set of full classes
        #train_set = get_subclass_dataset(train_set, classes=cls_list[P.one_class_idx])
        #test_set = get_subclass_dataset(test_set, classes=cls_list[P.one_class_idx])

    kwargs = {'pin_memory': False, 'num_workers': 4}
    '''
    if P.multi_gpu:
        train_sampler = DistributedSampler(train_set, num_replicas=P.n_gpus, rank=P.local_rank)
        test_sampler = DistributedSampler(test_set, num_replicas=P.n_gpus, rank=P.local_rank)
        train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=P.batch_size, **kwargs)
        test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=P.test_batch_size, **kwargs)
    else:
        train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
        test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
    '''
    '''
    if P.ood_dataset is None:
        if P.one_class_idx is not None:
            P.ood_dataset = list(range(P.n_superclasses))
            P.ood_dataset.pop(P.one_class_idx)
        elif P.dataset == 'cifar10':
            P.ood_dataset = ['svhn', 'lsun_resize', 'imagenet_resize', 'lsun_fix', 'imagenet_fix', 'cifar100', 'interp']
        elif P.dataset == 'imagenet':
            P.ood_dataset = ['food_101'] #['cub', 'stanford_dogs', 'flowers102']
    '''
    #ood_test_loader = dict()
    '''
    for ood in P.ood_dataset:
        if ood == 'interp':
            ood_test_loader[ood] = None  # dummy loader
            continue

        if P.one_class_idx is not None:
            ood_test_set = get_subclass_dataset(full_test_set, classes=cls_list[ood])
            ood = f'one_class_{ood}'  # change save name
        else:
            ood_test_set = get_dataset(P, dataset=ood, test_only=True, image_size=P.image_size)

        if P.multi_gpu:
            ood_sampler = DistributedSampler(ood_test_set, num_replicas=P.n_gpus, rank=P.local_rank)
            ood_test_loader[ood] = DataLoader(ood_test_set, sampler=ood_sampler, batch_size=P.test_batch_size, **kwargs)
        else:
            ood_test_loader[ood] = DataLoader(ood_test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
    '''
    #for ood in P.ood_dataset:
    #    ood_test_loader[ood] = ood_loader
    ### Initialize model ###

    simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)
    P.shift_trans, P.K_shift = C.get_shift_module(P, eval=True)
    P.shift_trans = P.shift_trans.to(device)

    model = C.get_classifier(P.model, n_classes=P.n_classes).to(device)
    model = C.get_shift_classifer(model, P.K_shift).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    if P.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
        lr_decay_gamma = 0.1
    #elif P.optimizer == 'lars':
    #    from torchlars import LARS
    #    base_optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    #    optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
    #    lr_decay_gamma = 0.1
    else:
        raise NotImplementedError()

    if P.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, P.epochs)
    elif P.lr_scheduler == 'step_decay':
        milestones = [int(0.5 * P.epochs), int(0.75 * P.epochs)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
    else:
        raise NotImplementedError()

    from training.scheduler import GradualWarmupScheduler
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10.0, total_epoch=P.warmup, after_scheduler=scheduler)

    if P.resume_path is not None:
        resume = True
        model_state, optim_state, config = load_checkpoint(P.resume_path, mode='last')
        model.load_state_dict(model_state, strict=not P.no_strict)
        optimizer.load_state_dict(optim_state)
        start_epoch = config['epoch']
        best = config['best']
        error = 100.0
    else:
        resume = False
        start_epoch = 1
        best = 100.0
        error = 100.0

    if P.mode == 'sup_linear' or P.mode == 'sup_CSI_linear':
        assert P.load_path is not None
        checkpoint = torch.load(P.load_path)
        model.load_state_dict(checkpoint, strict=not P.no_strict)

    if P.multi_gpu:
        simclr_aug = apex.parallel.DistributedDataParallel(simclr_aug, delay_allreduce=True)
        model = apex.parallel.convert_syncbn_model(model)
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

    if 'sup' in P.mode:
        from training.sup import setup
    else:
        from training.unsup import setup
    train, fname = setup(P.mode, P)

    logger = Logger(fname, ask=not resume, local_rank=P.local_rank)
    logger.log(P)
    logger.log(model)

    if P.multi_gpu:
        linear = model.module.linear
    else:
        linear = model.linear
    linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=P.weight_decay)

    eval_metrics = dict()
    eval_metrics['AUROC'] = []
    eval_metrics['AUPR'] = []
    eval_metrics['FPR95'] = []

    #if P.load_path is not None:
    #    checkpoint = torch.load(P.load_path)
    #    model.load_state_dict(checkpoint, strict=not P.no_strict)

    P.load_path = logger.logdir

    # Run experiments
    for itr in range(1, P.runs+1):
        print("Run ", itr)
        P.resize_factor = 0.08
        P.resize_fix  = False
        P.ood_layer = P.ood_layer[0]
        for epoch in range(start_epoch, P.epochs + 1):
            logger.log_dirname(f"Epoch {epoch}")
            model.train()

            if P.multi_gpu:
                train_sampler.set_epoch(epoch)

            kwargs = {}
            kwargs['linear'] = linear
            kwargs['linear_optim'] = linear_optim
            kwargs['simclr_aug'] = simclr_aug

            train(P, epoch, model, criterion, optimizer, scheduler_warmup, train_loader, logger=logger, **kwargs)

            #model.eval()

            if epoch % P.save_step == 0 and P.local_rank == 0:
                if P.multi_gpu:
                    save_states = model.module.state_dict()
                else:
                    save_states = model.state_dict()
                save_checkpoint(itr, epoch, save_states, optimizer.state_dict(), logger.logdir)
                save_linear_checkpoint(itr, linear_optim.state_dict(), logger.logdir)

            if epoch % P.error_step == 0: # and ('sup' in P.mode):
                auroc_dict, aupr_dict, fpr95_dict = evaluate(P, itr, ood_layer_eval)
                bests = []
                print('AUROC Metric: [ood_dataset split value]')
                for ood in auroc_dict.keys():
                    message = ''
                    best_auroc = 0
                    for ood_score, auroc in auroc_dict[ood].items():
                        eval_metrics['AUROC'].append(auroc)
                        message += '[%s %s %.4f] ' % (ood, ood_score, auroc)
                        if auroc > best_auroc:
                            best_auroc = auroc
                    message += '[%s %s %.4f] ' % (ood, 'best', best_auroc)
                    if P.print_score:
                        print(message)
                    bests.append(best_auroc)
                bests = map('{:.4f}'.format, bests)
                print('\t'.join(bests))

                print('AUPR Metric: [ood_dataset model value]')
                for ood in aupr_dict.keys():
                    message = ''
                    for ood_score, aupr in aupr_dict[ood].items():
                        eval_metrics['AUPR'].append(aupr)
                        message += '[%s %s %.4f] ' % (ood, ood_score, aupr)
                    if P.print_score:
                        print(message)

                print('FPR95 Metric: [ood_dataset model value]')
                for ood in fpr95_dict.keys():
                    message = ''
                    for ood_score, fpr in fpr95_dict[ood].items():
                        eval_metrics['FPR95'].append(fpr)
                        message += '[%s %s %.4f] ' % (ood, ood_score, fpr)
                    if P.print_score:
                        print(message)
        print()

    print("AUROC: {} +- {} | AUPR: {} +- {} | FPR95: {} +- {}".format(
                                                    np.mean(eval_metrics['AUROC']),
                                                    np.std(eval_metrics['AUROC']), 
                                                    np.mean(eval_metrics['AUPR']), 
                                                    np.std(eval_metrics['AUPR']), 
                                                    np.mean(eval_metrics['FPR95']),
                                                    np.std(eval_metrics['FPR95']))
                                                )

if __name__ == "__main__":
    main()