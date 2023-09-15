import torch.nn as nn
import time
from benchmark.utils import save_checkpoint, knn, evaluate_acc
from tqdm.notebook import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import torch.nn.functional as F
import torch
import os
from loguru import logger 

def train_one_epoch(
        model,
        device,
        dataloader,
        criterion,
        optimizer,
        lr_scheduler=None,
        epoch=0,
        args=None,
        do_float16 = True
):
    if do_float16:
        scaler = torch.cuda.amp.GradScaler()

    model.train()
    total_loss = 0

    for i, data in (pbar := tqdm(enumerate(dataloader), leave=False)):
        images, target = data[0], data[1].to(device)
        bsz = target.shape[0]

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if args.training_mode in ["SupCon", 'RotNet']:

                pred = model(images.to(device))
                loss = criterion(pred, target)


            elif args.training_mode == "SimCLR":
                images = torch.cat([images[0], images[1]], dim=0).to(device)

                features = model(images)
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                loss = criterion(features)
            else:
                raise ValueError("training mode not supported")

        optimizer.zero_grad()

        if do_float16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if lr_scheduler:
            lr_scheduler.step()

        total_loss += loss.item()

        pbar.set_description(f'Epoch {epoch} : Loss {total_loss / (i + 1) :.3f}')

def train(args, model, optimizer, criterion, lr_scheduler, device, train_loader, test_loader,  ):

    best_prec1 = 0


    if args.warmup:
        wamrup_epochs = 10
        logger.info(f"Warmup training for {wamrup_epochs} epochs")
        warmup_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.01,
            max_lr=args.lr,
            step_size_up=wamrup_epochs * len(train_loader),
        )
        for epoch in range(wamrup_epochs):
            train_one_epoch(
                model,
                device,
                train_loader,
                criterion,
                optimizer,
                warmup_lr_scheduler,
                epoch,
                args,
            )

    for epoch in range(0, args.epochs):

        train_one_epoch(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            lr_scheduler,
            epoch,
            args,
        )

        if args.training_mode in ["SupCon", ]:
            prec1, _ = evaluate_acc(model, device, test_loader)

        elif args.training_mode in ["SimCLR", 'RotNet']:
            prec1, _ = knn(model, device, test_loader)

        # remember best accuracy and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        d = {
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_prec1": best_prec1,
            "optimizer": optimizer.state_dict(),
        }

        save_checkpoint(
            d,
            is_best,
            os.path.join(args.result_sub_dir, "checkpoint"),
        )

        if not (epoch + 1) % args.save_freq:
            save_checkpoint(
                d,
                is_best,
                os.path.join(args.result_sub_dir, "checkpoint"),
                filename=f"checkpoint_{epoch + 1}.pth.tar",
            )

        logger.info(
            f"Epoch {epoch}, validation accuracy {prec1}, best_prec {best_prec1}"
        )

    return model


