from torchvision.datasets import CIFAR100
import torch
from torch.utils.data import ConcatDataset
from torchvision import transforms
from benchmark.datasets.datasets import OODDataset, TwoCropTransform
from torch.utils.data import DataLoader



def get_flowers_cifar100(transform):
    val = CIFAR100(root='.', download=True, train=True, transform=transform)

    def sparse2coarse(targets):
        """Convert Pytorch CIFAR100 sparse targets to coarse targets.

        Usage:
            trainset = torchvision.datasets.CIFAR100(path)
            trainset.targets = sparse2coarse(trainset.targets)
        """
        coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
        return coarse_labels[targets]

    val.targets = sparse2coarse(val.targets)

    flowers_idx = [idx for idx, target in enumerate(val.targets) if target in [2]]

    val = torch.utils.data.Subset(val, flowers_idx)

    return val


def get_dataloaders_cifar10_exp(dataset_name, args, batch_size=32, normalize=True, size=32, doCLR=False,
                                random_state=42, num_workers=16):
    assert dataset_name in ['cifar10', 'cifar10h']

    transform_train = [
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]

    if args.training_mode == "RotNet":
        transform_train = [
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]

    transform_test = [transforms.Resize([size, size]), transforms.ToTensor()]

    norm_layer = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
    )

    if normalize:
        transform_train.append(norm_layer)
        transform_test.append(norm_layer)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    if args.training_mode == "SimCLR" and doCLR:
        transform_train = TwoCropTransform(transform_train)

    train_set = OODDataset(dataset_name, split='Train', transform=transform_train,
                           rotnet=args.training_mode == "RotNet", random_state=random_state)
    test_set = OODDataset(dataset_name, split='Test', transform=transform_test, random_state=random_state)

    two_class_set = OODDataset(dataset_name, split='Out', transform=transform_test, random_state=random_state)
    flowers = get_flowers_cifar100(transform=transform_train)

    # the two classes
    if dataset_name == 'cifar10':
        ood_set = ConcatDataset([flowers, two_class_set])
    elif dataset_name == 'cifar10h':
        ood_set = flowers
        if not doCLR:
            test_set = ConcatDataset([test_set, two_class_set])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    ood_loader = DataLoader(
        ood_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader, ood_loader, train_set, test_set, ood_set


