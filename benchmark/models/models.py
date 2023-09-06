from benchmark.models.resnet import SupResNet, SSLResNet
from benchmark.losses.supcon import SupConLoss
import torch

def get_model(args, in_distro, device, train_loader):


    if args.training_mode in ["SupCon", 'RotNet'] and args.arch == "resnet50":

        num_classes = 4 if args.training_mode == 'RotNet' else len(in_distro)
        model = SupResNet(arch=args.arch, num_classes=num_classes).to(device)
        model.encoder = torch.nn.DataParallel(model.encoder).to(device)
        criterion = torch.nn.CrossEntropyLoss()

    elif args.training_mode == "SimCLR" and args.arch == "resnet50":
        model = SSLResNet(arch=args.arch).to(device)
        model.encoder = torch.nn.DataParallel(model.encoder).to(device)
        criterion = SupConLoss(temperature=args.temperature).cuda()

    else:
        raise KeyError(f'{args.training_mode} and {args.arch} is not a valid combination')

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    for p in optimizer.param_groups:
        p["lr"] = args.lr
        p["initial_lr"] = args.lr

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * len(train_loader), 1e-4
    )

    return model, criterion, optimizer, lr_scheduler