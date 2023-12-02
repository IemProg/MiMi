import os
import time
import argparse
import datetime
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.models as models

import collections.abc as container_abcs
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data.build import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import (
    load_checkpoint,
    save_checkpoint,
    save_checkpoint_best,
    get_grad_norm,
    auto_resume_helper,
    reduce_tensor
)

from drloc import cal_selfsupervised_loss, SymKlCriterion

# import EarlyStopping
from pytorchtools import EarlyStopping

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--dsets_type', type=str, help='path to dataset', default="decathlon")
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--exp_name', type=str, help='Experiment name with in output folder')
    parser.add_argument('--model_type', type=str, help='Model type')

    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O2', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--finetune', type=int, default=1, help='Finetunning mode')

    parser.add_argument('--OPT', type=str, default='adamw', choices=['SGD', 'adamw'], help='optimizer')
    parser.add_argument('--WD', type=float, default=0.05, help='weight decay')
    parser.add_argument('--LR', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--WU', type=int, default=20, help='warm-up epochs')
    parser.add_argument('--WU_LR', type=float, default=5e-7, help='warm-up epochs')
    parser.add_argument('--pretrained_model', type=str, help='Experiment name with in output folder')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')
    parser.add_argument("--use_drloc", action='store_true', help="Use Dense Relative localization loss")
    parser.add_argument("--drloc_mode", type=str, default="l1", choices=["l1", "ce", "cbr"])
    parser.add_argument("--lambda_drloc", type=float, default=0.5,
                        help="weight of Dense Relative localization loss")
    parser.add_argument("--sample_size", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--use_multiscale", action='store_true')
    parser.add_argument("--ape", action="store_true", help="using absolute position embedding")
    parser.add_argument("--rpe", action="store_false", help="using relative position embedding")
    parser.add_argument("--use_normal", action="store_true")
    parser.add_argument("--use_abs", action="store_true")
    parser.add_argument("--ssl_warmup_epochs", type=int, default=20)
    parser.add_argument("--total_epochs", type=int, default=100)

    parser.add_argument("--type_adapters", type=str, default="parallel")
    parser.add_argument("--size_adapters", type=int, default=32)
    parser.add_argument('--param_ratios', help='delimited list input', type=str)
    # parser.add_argument('--param_ratios', help='delimited list input', type=str, action='append', nargs='+')

    # Pruning
    parser.add_argument("--prune_layer", type=str, default="parallel_mlp")

    # PRUNING AMOUNT
    parser.add_argument("--prune_type", type=str, default="magnitude")
    parser.add_argument("--prune_struct", type=str, default="structured")
    parser.add_argument("--prune_amount", type=float, default=0.2)
    parser.add_argument("--delta_loss", type=float, default=0.1)
    parser.add_argument("--weighted", type=int, default=0, help="StructLAMP++")
    parser.add_argument("--scaling", type=int, default=1, help="StructLAMP++")
    parser.add_argument("--range", type=int, default=1)

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # Debugging mode
    parser.add_argument("--debug", default=False, help="Debugging Mode. Default = False.")

    args, unparsed = parser.parse_known_args()
    return args  # , config


def main(config):
    logger.info(f"\t\t Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    logger.info(f"\t\t FINETUNNING Argument: {config.MODEL.FINETUNE}")
    logger.info(f"\t\t DRLOC Argument: {config.TRAIN.USE_DRLOC}")
    current = os.getcwd()

    if config.MODEL.FINETUNE == 1:
        logger.info(f"\t\t Finetunning the model: {config.MODEL.TYPE}")
        if config.MODEL.TYPE == "vit":
            pre_model_path = os.path.join(current, "pretrained", config.MODEL.NAME + ".npz")
            model = build_model(config)
            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.hidden_size, 21843)
            model.head = classifier

            checkpoint = np.load(pre_model_path)
            msg = model.load_from(checkpoint)
            logger.info(f"Checkpoint for ViT-B/16 finetunning:{msg}")
            classifier = nn.Linear(model.hidden_size, config.MODEL.NUM_CLASSES)
            model.head = classifier
            model.requires_grad = True
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\t\t at Start: Number of params: {n_parameters}")
        elif config.MODEL.TYPE == "cvt":
            logger.info("\t\t Finetunning CvT-13 224x224")
            pre_model_path = os.path.join(current, "pretrained", "CvT-13-224x224-IN-1k" + ".pth")
            model = build_model(config)
            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.dim_embed, 1000)
            model.head = classifier
            model.head.requires_grad = True

            checkpoint = torch.load(pre_model_path, map_location='cpu')
            msg = model.load_state_dict(checkpoint, strict=True)
            logger.info(f"Checkpoint for CvT-13-224*224 finetunning")
            # Change it back to dataset's number of classes
            classifier = nn.Linear(model.dim_embed, config.MODEL.NUM_CLASSES)
            model.head = classifier
            model.requires_grad = True
        elif config.MODEL.TYPE == "swin":
            pre_model_path = os.path.join(current, "pretrained", config.MODEL.NAME + ".pth")
            model = build_model(config)
            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.num_features, 1000)
            model.head = classifier
            model.head.requires_grad = True

            checkpoint = torch.load(pre_model_path, map_location='cpu')

            msg = model.load_state_dict(checkpoint['model'], strict=True)
            #msg = model.load_state_dict(checkpoint, strict=False)

            # Change it back to dataset's number of classes
            classifier = nn.Linear(model.num_features, config.MODEL.NUM_CLASSES)
            model.head = classifier
            model.requires_grad = True
            for param in model.parameters():
                param.requires_grad = True
            logger.info("\t\t Finetunning {} Transformer".format(config.MODEL.TYPE))
        else:
            raise NotImplementedError("Finetunning for this model is not implemented yet")

    elif config.MODEL.FINETUNE == 3:
        if config.MODEL.TYPE == "swin_adapters" or config.MODEL.TYPE == "swin_adapters_layer":
            logger.info("\t\t Finetunning Swin Transformer Using Houlsbi Adapters")
            logger.info(f"\t\t TYPE OF ADAPTERS: {config.TRAIN.TYPE_ADAPTERS}")
            logger.info(f"\t\t SIZE OF ADAPTERS: {config.TRAIN.SIZE_ADAPTERS}")
            logger.info(f"\t\t STAGEs SIZE OF ADAPTERS: {config.MODEL.SWIN.PARAM_RATIOS}")

            pre_model_path = os.path.join(current, "pretrained", config.MODEL.NAME + ".pth")
            model = build_model(config)
            # Change it to ImageNet, number of classes
            classifier = nn.Linear(model.num_features, 1000)
            model.head = classifier

            checkpoint = torch.load(pre_model_path, map_location='cpu')
            # Upload all weights with same name -> Parallel
            msg = model.load_state_dict(checkpoint['model'], strict=False)

            # Change it back to dataset's number of classes
            classifier = nn.Linear(model.num_features, config.MODEL.NUM_CLASSES)
            model.head = classifier

            model.requires_grad = True
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\t\t at Start: Number of params: {n_parameters}")

            for p in model.parameters():
                p.requires_grad = False

            for name, param in model.named_parameters():
                if "parallel_mlp" in str(name):
                    param.requires_grad = True
                if "norm3" in str(name):
                    param.requires_grad = True
                if "norm4" in str(name):
                    param.requires_grad = True
                # Classifier params to True
                if ("head" in name):
                    param.requires_grad = True
    else:
        raise NotImplementedError("Finetunning for this model is not implemented yet")

    print("Inference time: ", inference_time(model), "ms")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")


def inference_time(model):
    device = torch.device("cuda")
    model.to(device)
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    logger.info(f"Mean :{mean_syn}")
    logger.info(f"std_syn :{std_syn}")
    return mean_syn


if __name__ == "__main__":
    # _, config = parse_option()
    args = parse_option()
    if not args.debug:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = args.local_rank

    print("\t\t args.local_rank: ", args.local_rank)
    datasets = ["flowers102"]

    root_path = os.getcwd()
    datasets_path = os.path.join(root_path, "datasets", str(args.dsets_type))
    output_folder = os.path.join(root_path, "output", args.exp_name)
    try:
        os.mkdir(output_folder)
    except:
        print("Output folder for {} exists already !".format(args.exp_name))

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    # Use args arguments
    if not args.debug:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()

    for dset in datasets:
        print("****************** Dataset: {} ******************".format(dset))
        # Dataset path
        if args.dsets_type in ["decathlon", "domainnet"]:
            args.data_path = os.path.join(datasets_path, dset)
        else:
            args.data_path = datasets_path
        args.output = os.path.join(output_folder, dset)
        args.dataset_name = dset
        try:
            os.mkdir(args.output)
        except:
            print("Output folder for {} exists already !".format(args.output))

        config = get_config(args)

        if config.AMP_OPT_LEVEL != "O0":
            assert amp is not None, "amp not installed!"

        if not config.DEBUG:
            seed = config.SEED + dist.get_rank()
        else:
            seed = config.SEED
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

        # linear scale the learning rate according to total batch size, may not be optimal
        if not config.DEBUG:
            linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
            linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
            linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        else:
            linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
            linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
            linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
        # gradient accumulation also need to scale the learning rate
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()

        os.makedirs(config.OUTPUT, exist_ok=True)
        if not config.DEBUG:
            logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
            if dist.get_rank() == 0:
                path = os.path.join(config.OUTPUT, "config.json")
                with open(path, "w") as f:
                    f.write(config.dump())
                logger.info(f"Full config saved to {path}")
        else:
            logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
            path = os.path.join(config.OUTPUT, "config.json")
            with open(path, "w") as f:
                f.write(config.dump())
            logger.info(f"Full config saved to {path}")

        # print config
        logger.info(config.dump())
        main(config)
        print("------------------------ Done Dataset: {} -------------------".format(dset))
