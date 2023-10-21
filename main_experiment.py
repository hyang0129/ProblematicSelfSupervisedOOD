from pathlib import Path
from benchmark.datasets.datasets import get_dataloaders
from benchmark.trainer import train
from benchmark.models.models import get_model
from benchmark.evaluation import run_evaluation
from loguru import logger
import sys
import click
from munch import Munch

sys.path.append("ProblematicSelfSupervisedOOD")


@click.command()
@click.option("--arch", default="resnet50", help="Architecture, so resnet50")
@click.option(
    "--dataset",
    required=True,
    help="One of icmlface, food101, cars, cifar10, or cifar100",
)
@click.option(
    "--normalize", default=True, help="Normalize data to imagenet std and mean"
)
@click.option(
    "--batch_size",
    default=256,
    type=int,
    help="Large batchsize recommended for self supervised learning",
)
@click.option(
    "--data_dir",
    default="data",
    help="Where to store data. You would need to place manually downloaded datasets here",
)
@click.option("--size", default=64, type=int, help="Image size going into network")
@click.option("--temperature", default=0.5, type=float, help="Temperature for SimCLR")
@click.option("--warmup", default=True, help="Do warm up training")
@click.option("--lr", default=0.5, help="learning rate max", type=float)
@click.option("--momentum", default=0.9, help="SGD momentum", type=float)
@click.option("--weight_decay", default=1e-4, help="SGD weight decay")
@click.option(
    "--epochs",
    default=0,
    type=int,
    help="We recommend 500 for self supervised and up to 150 for supervised, if 0 sets to 150 for supervised and 500 for self supervised",
)
@click.option(
    "--training_mode",
    required=True,
    help="SimCLR is Contrastive learning, SupCon is softmax CE supervised, and RotNet is rotation based",
)
@click.option("--print_freq", default=100, type=int)
@click.option("--save_freq", default=50, type=int)
@click.option(
    "--random_state",
    default=42,
    type=int,
    help="affects which classes are assigned as OOD and ID",
)
@click.option(
    "--results_dir",
    default="results",
    help="directory to put results, which will tree into results/dataset/method",
)
@click.option(
    "--device",
    default="cuda:0",
    help="This is meant to run on an A100 due to gpu memory requirements",
)
def main(
    arch,
    dataset,
    normalize,
    batch_size,
    data_dir,
    size,
    temperature,
    warmup,
    lr,
    momentum,
    weight_decay,
    epochs,
    training_mode,
    print_freq,
    save_freq,
    random_state,
    results_dir,
    device,
):
    args = Munch()

    args.arch = arch
    args.dataset = dataset
    args.normalize = normalize
    args.batch_size = batch_size
    args.data_dir = data_dir
    args.size = size
    args.temperature = temperature
    args.warmup = warmup
    args.lr = lr
    args.momentum = momentum
    args.weight_decay = weight_decay
    args.epochs = epochs
    args.training_mode = training_mode
    args.print_freq = print_freq
    args.save_freq = save_freq
    args.random_state = random_state
    args.results_dir = results_dir
    args.result_sub_dir = (
        args.results_dir
        + f"/{args.dataset}/{args.training_mode}/state_{args.random_state}"
    )
    args.best_checkpoint_path = args.result_sub_dir + "/checkpoint/model_best.pth.tar"
    args.log_path = args.result_sub_dir + "/logs.txt"
    args.device = device

    logger.add(args.log_path, level="INFO", backtrace=True)

    logger.info(args)

    args.clusters = 1  # always one and used in SimCLR for SSD method

    if epochs == 0 and args.training_mode == "SupCon":
        args.epochs = 150
    elif args.epochs == 0:
        args.epochs = 500

    Path(args.result_sub_dir + "/checkpoint").mkdir(parents=True, exist_ok=True)

    # do training

    (
        train_loader,
        test_loader,
        ood_loader,
        train_set,
        test_set,
        ood_set,
    ) = get_dataloaders(
        args.dataset,
        args,
        args.batch_size,
        size=args.size,
        doCLR=args.training_mode == "SimCLR",
        random_state=args.random_state,
        num_workers=16,
    )
    in_distro = train_set.in_distro

    model, criterion, optimizer, lr_scheduler = get_model(
        args, in_distro, args.device, train_loader
    )

    do_train = True

    if do_train:
        model = train(
            args,
            model,
            optimizer,
            criterion,
            lr_scheduler,
            args.device,
            train_loader,
            test_loader,
        )

    # do evaluation

    checkpoint_path = args.best_checkpoint_path

    (
        train_loader,
        test_loader,
        ood_loader,
        train_set,
        test_set,
        ood_set,
    ) = get_dataloaders(
        args.dataset,
        args,
        16,
        size=args.size,
        doCLR=args.training_mode == "False",
        random_state=args.random_state,
        num_workers=16,
    )

    run_evaluation(
        model,
        args,
        args.device,
        checkpoint_path,
        train_loader,
        test_loader,
        ood_loader,
    )


if __name__ == "__main__":
    main()
