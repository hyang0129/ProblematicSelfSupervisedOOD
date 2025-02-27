import os
import copy
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10 as cifar_10
from torchvision.datasets import CIFAR100 as cifar_100
import torchvision.datasets as dset
from classification import ViT_Classifier, load_model
from tuning_util import maybe_dictionarize, Places, Textures, ImageNet, iNaturalist, SUN
from tuning_cfg import parse_arguments
from sklearn import metrics
from sklearn.metrics import accuracy_score as Acc
from sklearn.metrics import roc_auc_score as Auc
from sklearn.metrics import roc_curve as Roc
from scipy import interpolate
from scipy.special import logsumexp
import numpy as np
import pandas as pd
import shutil
from adj_dataset import FaceDataset, CarsDataset, FoodDataset, Cifar100Dataset, Cifar10Dataset

to_np = lambda x: x.detach().cpu().numpy()
def max_logit_score(logits):
    return to_np(torch.max(logits, -1)[0])
def msp_score(logits):
    prob = torch.softmax(logits, -1)
    return to_np(torch.max(prob, -1)[0])
def energy_score(logits):
    return to_np(torch.logsumexp(logits, -1))

def infer(args, pth_dir, epoch, model_type='ViT-B-32', dataset_name = 'Face', seed = 0):
    pth_name = os.path.join("checkpoints", "epoch_" + str(epoch) + ".pt")
    pre_train = os.path.join(pth_dir, pth_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128

    if dataset_name == 'Face':
        ds_class = FaceDataset

    elif dataset_name == 'Cars':
        ds_class = CarsDataset

    elif dataset_name == 'Food':
        ds_class = FoodDataset

    else:
        raise NotImplementedError

    dataset = ds_class(seed = seed)

    print('loading models')
    vit_class, process_train, process_test = load_model(model_type=model_type, pre_train=pre_train, dataset=dataset, device=device)
    
    vit_class.fc_yes.requires_grad = False
    vit_class.fc_no.requires_grad = False

    print('prepping dataset')
    dataset = ds_class(preprocess_train = process_train, preprocess_test = process_test, batch_size = batch_size, seed = seed)
    test_dataset = {
        dataset_name: dataset.ood_loader,
    }

    test_loader = dataset.test_loader  

    model = vit_class.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
   
    id_lis_epoch, ood_lis_epoch = cal_all_metric(test_loader, model, epoch, test_dataset)
    
    return ood_lis_epoch
    
            
def cal_all_metric(id_dataset, model, epoch, ood_dataset=None, flag = True):
    model.eval()
    pred_lis = []
    gt_lis = []
    
    ind_logits, ind_prob, ind_energy = [], [], []
    if flag:
        ind_ctw, ind_atd = [], []
    res = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(id_dataset)):
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            labels = batch['labels'].cuda()
            logits, logits_no, _ = model(inputs)
            
            pred_lis += list(torch.argmax(logits, -1).detach().cpu().numpy())
            gt_lis += list(labels.detach().cpu().numpy())
            
            ind_logits += list(max_logit_score(logits))
            ind_prob += list(msp_score(logits))
            ind_energy += list(energy_score(logits))
            
            if flag:
                idex = torch.argmax(logits, -1).unsqueeze(-1)
                yesno = torch.cat([ logits.unsqueeze(-1), logits_no.unsqueeze(-1) ], -1)
                yesno = torch.softmax(yesno, dim=-1)[:,:,0]
                yesno_s = torch.gather(yesno, dim=1, index=idex)
                ind_ctw += list(yesno_s.detach().cpu().numpy())
                ind_atd += list((yesno * torch.softmax(logits, -1)).sum(1).detach().cpu().numpy())
                    
                
            
        for name, ood_data in ood_dataset.items():
            ood_logits, ood_prob, ood_energy = [], [], []
            if flag:
                ood_ctw, ood_atd = [], []
            for i, batch in tqdm(enumerate(ood_data)):
                batch = maybe_dictionarize(batch)
                inputs = batch["images"].cuda()
                labels = batch['labels'].cuda()
                logits, logits_no, _ = model(inputs)
                
                ood_logits += list(max_logit_score(logits))
                ood_prob += list(msp_score(logits))
                ood_energy += list(energy_score(logits))
            
                if flag:
                    idex = torch.argmax(logits, -1).unsqueeze(-1)
                    yesno = torch.cat([ logits.unsqueeze(-1), logits_no.unsqueeze(-1) ], -1)
                    yesno = torch.softmax(yesno, dim=-1)[:,:,0]
                    yesno_s = torch.gather(yesno, dim=1, index=idex)

                    ood_ctw += list(yesno_s.detach().cpu().numpy())
                    ood_atd += list((yesno * torch.softmax(logits, -1) ).sum(1).detach().cpu().numpy())
                    
                 
            #### MSP
            auc, fpr, aupr = cal_auc_fpr(ind_prob, ood_prob)
            res.append([epoch, "MSP", name, auc, fpr, aupr])
            #### MaxLogit
            auc, fpr, aupr = cal_auc_fpr(ind_logits, ood_logits)
            res.append([epoch, "MaxLogit", name, auc, fpr, aupr])
            #### Energy
            auc, fpr, aupr = cal_auc_fpr(ind_energy, ood_energy)
            res.append([epoch, "Energy", name, auc, fpr, aupr])
            if flag:
                auc, fpr, aupr = cal_auc_fpr(ind_ctw, ood_ctw)
                res.append([epoch, "CTW", name, auc, fpr, aupr])
                
                auc, fpr, aupr = cal_auc_fpr(ind_atd, ood_atd)
                res.append([epoch, "ATD", name, auc, fpr, aupr])
                
            
    pred_lis = np.array(pred_lis)
    gt_lis = np.array(gt_lis)
    acc = Acc(gt_lis, pred_lis)
    
    id_lis_epoch = [[epoch, acc]]
    ood_lis_epoch = res
    print(id_lis_epoch)
    for lis in ood_lis_epoch:
        print(lis)
    return id_lis_epoch, ood_lis_epoch
def cal_auc_fpr(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))
    auroc = metrics.roc_auc_score(ind_indicator, conf)
    fpr,tpr,thresh = Roc(ind_indicator, conf, pos_label=1)
    fpr = float(interpolate.interp1d(tpr, fpr)(0.95))

    aupr = metrics.average_precision_score(ind_indicator, conf)

    return auroc, fpr, aupr

def cal_fpr_recall(ind_conf, ood_conf, tpr=0.95):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))
    fpr,tpr,thresh = Roc(ind_indicator, conf, pos_label=1)
    fpr = float(interpolate.interp1d(tpr, fpr)(0.95))
    return fpr, thresh
    



if __name__ == '__main__':
    args = parse_arguments()


    print('starting evaluation')

    pth_dir = '.'
    
    header_ood = ['epoch', 'method', 'oodset', 'AUROC', 'FPR@95', 'AUPR']
    ood_lis = []

    model_type = "ViT-B-16"

    for i in range(10, 11):    ### evaluate the model of the 10-th epoch.
        ood_lis += infer(args, pth_dir, i, model_type=model_type, dataset_name = args.adj_dataset, seed = args.seed)

    df = pd.DataFrame(ood_lis, columns=header_ood)
    df.to_csv(os.path.join(pth_dir, 'ood_metric_.csv'), index=False)